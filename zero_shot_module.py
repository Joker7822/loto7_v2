from sentence_transformers import SentenceTransformer, util

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

def encode_numbers(numbers):
    return sbert_model.encode(" ".join(map(str, sorted(numbers))))

def most_similar(candidate, history, k=5):
    emb = encode_numbers(candidate)
    sims = [util.cos_sim(emb, encode_numbers(h))[0][0].item() for h in history]
    top = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
    return [history[i] for i in top]