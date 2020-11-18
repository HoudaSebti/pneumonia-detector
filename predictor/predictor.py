from predictor import model_generator

if __name__ == '__main__':
    model=model_generator.build_model((224,224,3))
    model.summary()