# need to import this for the function to work: from datasets import load_dataset

def download_FGVC():
    dataset_train = load_dataset("Multimodal-Fatima/FGVC_Aircraft_train")
    dataset_test = load_dataset("Multimodal-Fatima/FGVC_Aircraft_test")
    embeddings_test = load_dataset("Multimodal-Fatima/FGVC_Aircraft_test_embeddings")
    embeddings_train = load_dataset("Multimodal-Fatima/FGVC_Aircraft_test_embeddings")
    return dataset_train, dataset_test, embeddings_train, embeddings_test