from calidad_pqrs.utils import load_directory, load_data, drop_data, mapping_data, clean_text_TfIdf


def preprocessing_process():

    paths = load_directory(directory='Train')
    data = load_data(paths)

    data = drop_data(data)
    data = mapping_data(data)

    data = clean_text_TfIdf(data)

    return data



if __name__ == "__main__":

    data_procesada = preprocessing_process()
    print(data_procesada.head(4))