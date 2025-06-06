from datasets import load_dataset

def download_FGVC():
    
    dataset_train = load_dataset("Multimodal-Fatima/FGVC_Aircraft_train")
    dataset_test = load_dataset("Multimodal-Fatima/FGVC_Aircraft_test")
    embeddings_test = load_dataset("Multimodal-Fatima/FGVC_Aircraft_test_embeddings")
    embeddings_train = load_dataset("Multimodal-Fatima/FGVC_Aircraft_test_embeddings")
    return dataset_train["train"], dataset_test["test"], embeddings_train["openai_clip_vit_large_patch14"], embeddings_test