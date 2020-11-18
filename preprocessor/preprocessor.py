from preprocessor import dataset_generator

if __name__ == '__main__':

    train_images, train_labels = dataset_generator.generate_dataset(
        dataset_generator.Dataset_type.TRAIN,
        dataset_generator.generate_augmentation_sequence(
            [
                'Fliplr',
                'Affine',
                'Multiply'
            ],
            [
                {},
                {
                    'rotate' : 20
                },
                {
                    'mul' : (1.2, 1.5)
                }
            ]
        )
    )
    batches = dataset_generator.get_batches_generator(
        train_images,
        train_labels,
        batch_size=16
    )
    for batch in batches:
        print(batch[1])
