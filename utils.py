
from operator import itemgetter


def get_cropped_neuron(binary_mask, threshold=10):

    # find min and max values
    index_list = []
    for x in range(500):
        for y in range(500):
            pixel = binary_mask[x, y]
            if pixel:
                index_list.append((x, y))
            else:
                continue

    min_x = min(index_list, key=itemgetter(0))[0]
    min_y = min(index_list, key=itemgetter(1))[1]
    max_x = max(index_list, key=itemgetter(0))[0]
    max_y = max(index_list, key=itemgetter(1))[1]

    x_start = min_x - threshold
    if x_start < 0:
        x_start = 0
    x_end = max_x + threshold
    if x_end > 500:
        x_end = 500

    y_start = min_y - threshold
    if y_start < 0:
        y_start = 0
    y_end = max_y + threshold
    if y_end > 500:
        y_end = 500
    neuron_cropped = binary_mask[x_start: x_end, y_start: y_end]

    return neuron_cropped, x_start, x_end, y_start, y_end


