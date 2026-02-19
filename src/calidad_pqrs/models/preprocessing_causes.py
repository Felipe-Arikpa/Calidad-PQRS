from calidad_pqrs.utils import define_f3, load_directory, load_data, drop_data, mapping_data, define_service, define_f3, clean_text_TfIdf


def preprocessing_causes():

    paths = load_directory(directory='Train')
    data = load_data(paths)

    data = drop_data(data)
    data = mapping_data(data)

    data['Filtro 4_map'] = data.apply(define_service, axis=1)
    data['Filtro 3_map'] = data['Filtro 3'].apply(define_f3)

    data = clean_text_TfIdf(data)

    return data



if __name__ == "__main__":

    data_procesada = preprocessing_causes()
    print(data_procesada.head(4))