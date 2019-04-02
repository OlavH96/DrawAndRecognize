import tkinter as tk
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as ski_io
from skimage import io
from skimage.transform import resize


class App:

    def __init__(self, root: tk.Tk, model):
        self.root = root
        self.model = model
        self.canvas_width = 500
        self.canvas_height = 500
        root.title("MNIST Drawing App")
        self.canvas = tk.Canvas(width=self.canvas_height, height=self.canvas_width,bg="black")
        self.root.bind("<B1-Motion>", self.paint)
        self.canvas.pack()
        self.canvas.create_rectangle(0,0,self.canvas_width, self.canvas_height,fill="black")

        self.output_text_var = tk.StringVar()
        self.output_text = tk.Label(textvariable=self.output_text_var)
        self.output_text.pack()

        self.clear_canvas_button = tk.Button(text="Clear", command=self.clear_canvas)
        self.clear_canvas_button.pack()

        self.predict_letter_button = tk.Button(text="Predict", command=self.predict)
        self.predict_letter_button.pack()

    def paint(self, event):
        bg = "#FFFFFF"
        size = 20
        x1, y1 = (event.x - size), (event.y - size)
        x2, y2 = (event.x + size), (event.y + size)
        self.canvas.create_oval(x1, y1, x2, y2, fill=bg)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0,0,self.canvas_width, self.canvas_height,fill="black")


    def predict(self):
        self.canvas.postscript(file="tmp_canvas.eps",
                               colormode="gray",
                               width=self.canvas_width,
                               height=self.canvas_height,
                               pagewidth=self.canvas_width - 1,
                               pageheight=self.canvas_height - 1)

        data = ski_io.imread("tmp_canvas.eps")

        ski_io.imsave("canvas_image.png", data)
        img = io.imread('canvas_image.png', as_gray=True)
        img = resize(img, (28,28))

        print(self.model)
        data = np.asarray(img, dtype="float64")
        print(data.shape)

        #data /= 255
        data[data>0.01]=1
        data[data<0.01]=0

        plt.imshow(data)
        plt.show()
        print(data.shape)
        print(data)

        prediction = self.model.predict(data.reshape(1, 28, 28, 1))
        self.output_text_var.set(prediction.argmax())

if __name__ == '__main__':
    print(Path(".").absolute())
    model_path = Path(".").absolute().parent.parent / "models" / "model.h5"
    model = keras.models.load_model(model_path.absolute().as_posix())

    window = tk.Tk()
    app = App(window, model)
    window.mainloop()
