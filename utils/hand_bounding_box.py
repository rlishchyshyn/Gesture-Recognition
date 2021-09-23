# function to get the bounding box over the hand
def get_bounding_box(landmark_list):
    min_x_lm = landmark_list.copy()
    min_y_lm = landmark_list.copy()

    min_x_lm.sort(key=lambda l: l[0])
    min_y_lm.sort(key=lambda l: l[1])

    return min_x_lm[0][0], min_y_lm[0][1], min_x_lm[-1][0] - min_x_lm[0][0], min_y_lm[-1][1] - min_y_lm[0][1]
