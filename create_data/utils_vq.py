def vq_ids_to_string(ids, prefix="<VQ_", suffix=">"):
    return " ".join(f"{prefix}{int(i)}{suffix}" for i in ids)
