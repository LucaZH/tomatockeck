import tensorflow as tf
import numpy as np
import os

class TomatoChecker:
    def __init__(self, model_path, data_cat, img_width, img_height):
        self.model_path = model_path
        self.data_cat = data_cat
        self.img_width = img_width
        self.img_height = img_height
        self.model = self.load_model()

    def load_model(self):
        model = tf.keras.models.load_model(self.model_path)
        return model

    def predict_tomato_type(self, image):
        image_load = tf.keras.utils.load_img(image, target_size=(self.img_height, self.img_width))
        img_arr = tf.keras.utils.array_to_img(image_load)
        img_bat = tf.expand_dims(img_arr, 0)
        predict = self.model.predict(img_bat)
        score = tf.nn.softmax(predict)
        result = {label: round(score[0][i].numpy(), 4) for i, label in enumerate(self.data_cat)}
        return result

# data_cat = ['tomatoes_fresh', 'tomatoes_fresh_medium', 'tomatoes_rotten']
# img_width = 180
# img_height = 180
# model_path = os.path.join("model", "tomatoes.keras")

# tomato_checker = TomatoChecker(model_path, data_cat, img_width, img_height)
# tomato_checker = TomatoChecker(
#     model_path=os.path.join("model", "tomatoes.keras"),
#     data_cat=['tomatoes_fresh', 'tomatoes_fresh_medium', 'tomatoes_rotten'],
#     img_width=180,
#     img_height=180
# )
# print(tomato_checker.predict_tomato_type("dataset/test/test_4.png"))
