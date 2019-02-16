from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


def vis_model(model, show_shapes=False, show_layer_names=True, rankdir='TB'):
    """Visualize model in a notebook."""
    return SVG(
        model_to_dot(
            model, show_shapes=show_shapes, show_layer_names=show_layer_names, rankdir=rankdir
        ).create(prog='dot', format='svg')
    )
