import pygame
import sys
import numpy as np
import tensorflow as tf
from TensorFlowNN import NeuralNetwork

# some project constants
__PIXEL_SIZE = (14, 14)
__DIGIT_SIZE = 28
__WINDOW_SIZE = [x * __DIGIT_SIZE for x in __PIXEL_SIZE]

__NN_OUTPUT_SIZE = 18
__WINDOW_SIZE[0] += __PIXEL_SIZE[0] * __NN_OUTPUT_SIZE
__BAR_SIZE = (__NN_OUTPUT_SIZE - 11) * __DIGIT_SIZE

# initializing pygame
pygame.init()
surface = pygame.display.set_mode(__WINDOW_SIZE)
pygame.display.set_caption("Digit recognition")
clock = pygame.time.Clock()
pygame.font.init()

proj_font = pygame.font.SysFont('Arial', 16)
text_color = (100, 100, 100)
bar_color = (50, 0, 250)
display = np.zeros((28, 28))
dragPaint = False
pressedButton = 0
neural_output = np.zeros((1, 10), np.float32)

def brush_print(value: float, x: int, y: int):
    #firstly check if middle point is in drawing panel
    if 0 > x or x >= __DIGIT_SIZE:
        return

    if 0 > y or y >= __DIGIT_SIZE:
        return

    #define points
    points = [(x, y)]
    for temp in [-1, 1]:
        point_x = temp + x
        point_y = temp + y

        # add point if is in bounds
        if 0 <= point_x < __DIGIT_SIZE:
            points.append((point_x, y))

        if 0 <= point_y < __DIGIT_SIZE:
            points.append((x, point_y))

    for point_y, point_x in points:
        display[point_x][point_y] += value

        if display[point_x][point_y] > 1:
            display[point_x][point_y] = 1

        if display[point_x][point_y] < 0:
            display[point_x][point_y] = 0


def print_neural_output():
    # width of this screen is (9 x 16px, 2x16px) and starts at x = 29 * 16
    node_height = __WINDOW_SIZE[1]/10.
    #half of font size for 16 in pixels
    font_size_px = 11
    x_draw = (__DIGIT_SIZE + 1)* __PIXEL_SIZE[0]
    y_draw = node_height / 2. - font_size_px
    x_bar_draw = (__DIGIT_SIZE + 3) * __PIXEL_SIZE[0]

    bar = pygame.Rect(x_bar_draw, y_draw, __BAR_SIZE, font_size_px * 2)

    for i in range(10):
        text = proj_font.render(str(i) + ".", True, text_color)
        bar.width = __BAR_SIZE * neural_output[0][i]
        pygame.draw.rect(surface, bar_color, bar)
        surface.blit(text, (x_draw, y_draw))
        bar.y += node_height
        y_draw += node_height


if __name__ == "__main__":
    # loading neuralnetwork
    network = NeuralNetwork.load("./data/image_recog.h5")

    while True:
        clock.tick(60)


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                dragPaint = True
                pressedButton = event.button
            elif event.type == pygame.MOUSEBUTTONUP:
                dragPaint = False
                pressedButton = 0
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    #guess number
                    neural_output = network.guess(np.expand_dims(display, 0))
                    print("I think that it is: " + str(neural_output) + "\n")
                    print(np.argmax(neural_output))

                    display = np.zeros((28, 28))




        if dragPaint:
            # getting position and calculating pos to
            mousePos = pygame.mouse.get_pos()
            mapPos = [int(pos / __PIXEL_SIZE[0]) for pos in mousePos]
            # checking which LMB was clicked
            if pressedButton == 1:
                brush_print(0.5, *mapPos)

            if pressedButton == 2:
                brush_print(-0.7, *mapPos)

        # printing section
        surface.fill((0, 0, 0))

        rectangle = pygame.Rect(0, 0, __PIXEL_SIZE[0], __PIXEL_SIZE[1])
        for y, row in enumerate(display):
            rectangle.y = y * __PIXEL_SIZE[0]
            for x, value in enumerate(row):
                pygame.draw.rect(surface, [255 * value for _ in range(3)], rectangle)
                rectangle.x += __PIXEL_SIZE[0]
            rectangle.x = 0

        print_neural_output()
        pygame.display.flip()
        pygame.event.pump()


