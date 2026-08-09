import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import activations

__all__ = [
    "RelationAwareMessageLayer",
    "DisentangledEvidenceEncoder",
    "PriorEvidenceEncoder",
]


class RelationAwareMessageLayer(nn.Module):
    """
    A lightweight relation-aware aggregation layer implemented in pure PyTorch.
    It replaces the original DGL-based RelGraphConv so the project can run
    without DGL.
    """

    def __init__(self, in_dim, out_dim, num_rels, dropout=0.1):
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim, bias=False)
        self.msg_linear = nn.Linear(in_dim, out_dim, bias=False)
        self.rel_embeddings = nn.Embedding(num_rels, in_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.self_linear.weight)
        nn.init.xavier_uniform_(self.msg_linear.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)

    def forward(self, node_feats: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """
        node_feats: [N, D]
        edges: [E, 3] -> (src, rel, dst) over local node indices
        """
        out = self.self_linear(node_feats)

        if edges.numel() == 0:
            return out

        src = edges[:, 0].long()
        rel = edges[:, 1].long()
        dst = edges[:, 2].long()

        src_feat = node_feats[src]                           # [E, D]
        rel_feat = self.rel_embeddings(rel)                 # [E, D]
        msg = self.msg_linear(src_feat + rel_feat)          # [E, H]

        agg = torch.zeros_like(out)
        agg.index_add_(0, dst, msg)

        deg = torch.zeros(node_feats.size(0), device=node_feats.device, dtype=node_feats.dtype)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=node_feats.device, dtype=node_feats.dtype))
        deg = deg.clamp(min=1.0).unsqueeze(-1)

        out = out + agg / deg
        return self.dropout(out)


class DisentangledEvidenceEncoder(nn.Module):
    """
    Per-component graph encoder. Each component has its own input projection,
    then passes through several relation-aware aggregation layers.
    """

    def __init__(self, input_size, hidden_size, num_rels, num_hidden_layers=1, num_components=4, dropout=0.1):
        super().__init__()
        self.num_components = num_components
        self.dropout = nn.Dropout(dropout)

        self.component_projections = nn.ModuleList(
            [nn.Linear(input_size, input_size, bias=False) for _ in range(num_components)]
        )

        self.layers = nn.ModuleList()
        for _ in range(num_components):
            comp_layers = nn.ModuleList()
            if num_hidden_layers <= 0:
                comp_layers.append(RelationAwareMessageLayer(input_size, input_size, num_rels, dropout=dropout))
            else:
                comp_layers.append(RelationAwareMessageLayer(input_size, hidden_size, num_rels, dropout=dropout))
                for _ in range(num_hidden_layers - 1):
                    comp_layers.append(RelationAwareMessageLayer(hidden_size, hidden_size, num_rels, dropout=dropout))
                comp_layers.append(RelationAwareMessageLayer(hidden_size, input_size, num_rels, dropout=dropout))
            self.layers.append(comp_layers)

        for proj in self.component_projections:
            nn.init.xavier_uniform_(proj.weight)

    def forward(self, base_node_feats: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """
        base_node_feats: [N, D]
        edges: [E, 3]
        return: [N, C, D]
        """
        comp_outputs = []

        for comp_id in range(self.num_components):
            x = self.component_projections[comp_id](base_node_feats)
            x = self.dropout(x)

            for layer in self.layers[comp_id][:-1]:
                x = F.relu(layer(x, edges))
            x = self.layers[comp_id][-1](x, edges)

            comp_outputs.append(x.unsqueeze(1))  # [N, 1, D]

        return torch.cat(comp_outputs, dim=1)    # [N, C, D]


class PriorEvidenceEncoder(nn.Module):
    """
    PPENet prior-evidence module described in the revised manuscript.

    It combines a frozen graph-wide relational prior with disentangled,
    relation-aware evidence components and maps the aligned representation to
    the language-model hidden space.
    """

    def __init__(
        self,
        kge_embedding,
        input_size,
        num_rels,
        evidence_hidden_dim,
        evidence_num_hidden_layers,
        adapter_size,
        output_size=4096,
        hidden_act="silu",
        num_components=4,
        mi_weight=0.01,
        component_dropout=0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_components = num_components
        self.mi_weight = mi_weight

        self.ent_embeddings = nn.Embedding.from_pretrained(kge_embedding, freeze=True)
        self.rel_embeddings = nn.Embedding(num_rels, input_size)

        self.encoder = DisentangledEvidenceEncoder(
            input_size=input_size,
            hidden_size=evidence_hidden_dim,
            num_rels=num_rels,
            num_hidden_layers=evidence_num_hidden_layers,
            num_components=num_components,
            dropout=component_dropout,
        )

        self.adapter = nn.Sequential(
            nn.Linear(in_features=2 * input_size, out_features=adapter_size, bias=False),
            activations.ACT2FN[hidden_act],
            nn.Dropout(p=component_dropout),
            nn.Linear(in_features=adapter_size, out_features=output_size, bias=False),
        )

        for layer in self.adapter:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)

    def _fallback_components(self, base_vec: torch.Tensor) -> torch.Tensor:
        """
        base_vec: [N, D]
        return: [N, C, D]
        """
        return base_vec.unsqueeze(1).repeat(1, self.num_components, 1)

    def _component_independence_loss(self, comp_tensor: torch.Tensor) -> torch.Tensor:
        """
        comp_tensor: [M, C, D]
        MI-style surrogate:
        minimize off-diagonal cosine similarity^2 between components.
        """
        if comp_tensor.numel() == 0:
            return comp_tensor.new_tensor(0.0)

        x = F.normalize(comp_tensor, dim=-1)                 # [M, C, D]
        sim = torch.matmul(x, x.transpose(1, 2))            # [M, C, C]
        eye = torch.eye(self.num_components, device=sim.device, dtype=torch.bool).unsqueeze(0)
        off_diag = sim.masked_select(~eye)
        if off_diag.numel() == 0:
            return sim.new_tensor(0.0)
        return (off_diag ** 2).mean()

    def _build_local_graph(self, edges_list, device):
        """
        edges_list: [[h, r, t], ...] using global entity ids
        return:
            node_ids_sub: np.ndarray of original global node ids
            edges_local: torch.LongTensor [E, 3] over local node indices
        """
        if edges_list is None or len(edges_list) == 0:
            return None, None

        edges_arr = np.array(edges_list, dtype=np.int64)
        if edges_arr.ndim != 2 or edges_arr.shape[1] != 3:
            return None, None

        src = edges_arr[:, 0]
        rel = edges_arr[:, 1]
        dst = edges_arr[:, 2]

        node_ids_sub = np.unique(np.concatenate([src, dst]))
        node_id_to_idx = {old: idx for idx, old in enumerate(node_ids_sub)}

        mapped_src = np.array([node_id_to_idx[s] for s in src], dtype=np.int64)
        mapped_dst = np.array([node_id_to_idx[d] for d in dst], dtype=np.int64)

        edges_local = torch.tensor(
            np.stack([mapped_src, rel, mapped_dst], axis=1),
            dtype=torch.long,
            device=device,
        )
        return node_ids_sub, edges_local

    def _relation_aware_fusion(self, comp_tensor: torch.Tensor, rel_embed: torch.Tensor):
        """
        comp_tensor:
            query side  -> [B, C, D]
            entity side -> [B, K, C, D]
        rel_embed: [B, D]
        return fused tensors and beta
        """
        if comp_tensor.dim() == 3:
            # [B, C, D]
            scores = torch.einsum("bcd,bd->bc", comp_tensor, rel_embed)
            beta = F.softmax(scores, dim=-1)
            fused = (comp_tensor * beta.unsqueeze(-1)).sum(dim=1)
            return fused, beta

        if comp_tensor.dim() == 4:
            # [B, K, C, D]
            scores = torch.einsum("bkcd,bd->bkc", comp_tensor, rel_embed)
            beta = F.softmax(scores, dim=-1)
            fused = (comp_tensor * beta.unsqueeze(-1)).sum(dim=2)
            return fused, beta

        raise ValueError(f"Unsupported comp_tensor dim: {comp_tensor.dim()}")

    def forward(self, query_ids, relation_ids, entity_ids, subgraph=None, return_aux=False):
        device = query_ids.device
        batch_size = query_ids.size(0)
        K = entity_ids.size(1)

        q_global = self.ent_embeddings(query_ids)    # [B, D]
        e_global = self.ent_embeddings(entity_ids)   # [B, K, D]
        r_embed = self.rel_embeddings(relation_ids)  # [B, D]

        q_local_comp_all = []
        e_local_comp_all = []

        for i in range(batch_size):
            q_id = query_ids[i].item()
            ent_list = entity_ids[i]
            edges_list = None if subgraph is None else subgraph[i]

            # too small -> fallback
            if edges_list is None or len(edges_list) <= 10:
                q_local_comp = self._fallback_components(q_global[i].unsqueeze(0)).squeeze(0)  # [C, D]
                e_local_comp = self._fallback_components(e_global[i])                            # [K, C, D]
                q_local_comp_all.append(q_local_comp.unsqueeze(0))
                e_local_comp_all.append(e_local_comp.unsqueeze(0))
                continue

            node_ids_sub, edges_local = self._build_local_graph(edges_list, device=device)
            if node_ids_sub is None or edges_local is None or len(node_ids_sub) == 0:
                q_local_comp = self._fallback_components(q_global[i].unsqueeze(0)).squeeze(0)
                e_local_comp = self._fallback_components(e_global[i])
                q_local_comp_all.append(q_local_comp.unsqueeze(0))
                e_local_comp_all.append(e_local_comp.unsqueeze(0))
                continue

            node_id_to_idx = {old: idx for idx, old in enumerate(node_ids_sub)}
            node_ids_sub_t = torch.tensor(node_ids_sub, dtype=torch.long, device=device)
            base_emb_sub = self.ent_embeddings(node_ids_sub_t)                       # [N, D]
            local_comp_sub = self.encoder(base_emb_sub, edges_local)                 # [N, C, D]

            # query
            if q_id in node_id_to_idx:
                q_idx = node_id_to_idx[q_id]
                q_local_comp = local_comp_sub[q_idx]                                 # [C, D]
            else:
                q_local_comp = self._fallback_components(q_global[i].unsqueeze(0)).squeeze(0)

            # candidates
            e_comp_list = []
            for e_id in ent_list:
                eid_int = e_id.item()
                if eid_int in node_id_to_idx:
                    e_idx = node_id_to_idx[eid_int]
                    e_comp = local_comp_sub[e_idx]                                   # [C, D]
                else:
                    base_e = self.ent_embeddings(e_id.view(-1)).squeeze(0).unsqueeze(0)  # [1, D]
                    e_comp = self._fallback_components(base_e).squeeze(0)                # [C, D]
                e_comp_list.append(e_comp.unsqueeze(0))

            e_local_comp = torch.cat(e_comp_list, dim=0)                             # [K, C, D]
            q_local_comp_all.append(q_local_comp.unsqueeze(0))
            e_local_comp_all.append(e_local_comp.unsqueeze(0))

        q_local_comp = torch.cat(q_local_comp_all, dim=0)      # [B, C, D]
        e_local_comp = torch.cat(e_local_comp_all, dim=0)      # [B, K, C, D]

        q_local_fused, q_beta = self._relation_aware_fusion(q_local_comp, r_embed)   # [B, D], [B, C]
        e_local_fused, e_beta = self._relation_aware_fusion(e_local_comp, r_embed)   # [B, K, D], [B, K, C]

        q_concat = torch.cat([q_global, q_local_fused], dim=-1)      # [B, 2D]
        e_concat = torch.cat([e_global, e_local_fused], dim=-1)      # [B, K, 2D]

        q_out = self.adapter(q_concat)                                # [B, H]
        e_out = self.adapter(e_concat)                                # [B, K, H]
        e_out_flat = e_out.reshape(batch_size * K, -1)                # [B*K, H]

        mi_loss_q = self._component_independence_loss(q_local_comp)   # scalar
        mi_loss_e = self._component_independence_loss(
            e_local_comp.reshape(batch_size * K, self.num_components, self.input_size)
        )
        mi_loss = 0.5 * (mi_loss_q + mi_loss_e)

        if return_aux:
            aux = {
                "mi_loss": mi_loss,
                "q_local_comp": q_local_comp,
                "e_local_comp": e_local_comp,
                "q_beta": q_beta,
                "e_beta": e_beta,
            }
            return q_out, e_out_flat, aux

        return q_out, e_out_flat
