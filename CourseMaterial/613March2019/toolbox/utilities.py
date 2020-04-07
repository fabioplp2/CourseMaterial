from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

def image_plot(image, digit, cmap='bone_r', figsize=(5,5)):

    fig, ax = plt.subplots(1,1,figsize=figsize)
    ax.text(0.1,0.85,'Label : {}'.format(digit), fontsize=15, color ='b', transform=ax.transAxes)
    _ = ax.imshow(image, cmap='bone_r')


def transform_label(x): 
    ''' Using the one hot concept to model the 9 discrete possible integer values for classification to get
        probability of each class. Using argmax function allows us to get the correct class i.e. 0, 1, 2, ... 9
    '''
    one_hot = []
    for item in x:
        one_hot.append([float(int(i==item)) for i in range(0,10)])
    return np.array(one_hot)

def flatten_images(image):
    '''
    Flatten the image
    '''
    original_shape = image.shape # (number_images by y_pixels(rows) by x_pixels(columns))
    return image.reshape(original_shape[0], original_shape[1]*original_shape[1])

def mnist_weight_check(weight_tensor, figsize=(20,4), shape=[28,28], cmap='jet'):

    fig, mainax = plt.subplots(2,5,figsize=figsize)
    for i, ax in enumerate(mainax.flatten()):
        _ = ax.imshow(weight_tensor[:,i].reshape(shape[0],shape[1]), cmap=cmap)

## Conda install ffmpeg, 
## from matplotlib import animation, rc
## rc('animation', html='html5')
## show anim in another cell
def mnist_result_anim(images, digits_true, digits_predicted, shape=[28,28], cmap='bone_r', interval=500, repeat=True, n_iterations = 100):

    try:
        _ = digits_true.shape[1]
    except:
        digits_true = transform_label(digits_true)

    fig, ax = plt.subplots(1,1,figsize=(6,6))
    image = ax.imshow(np.zeros([28,28]), cmap=cmap)
    true_digit = ax.text(0.1, 0.9, s='True Value:', fontsize=12, transform = ax.transAxes, color='b')
    predicted_digit = ax.text(0.1, 0.85, s='PredictedValue:', fontsize=12, transform = ax.transAxes, color='g')

    def update(i):
        image=ax.imshow(images[i].reshape(shape[0],shape[1]),cmap=cmap)
        true_digit.set_text('True Value: {}'.format(np.argmax(digits_true[i])))
        predicted_digit.set_text('PredictedValue: {}'.format(np.argmax(digits_predicted[i])))
        color = '{}'.format('g' if np.argmax(digits_true[i]) == np.argmax(digits_predicted[i]) else 'r')
        predicted_digit.set_color(color)
        return image, true_digit, predicted_digit,
    if n_iterations is None:
        n_iterations = len(images)
    anim = animation.FuncAnimation(fig, update, frames=n_iterations, interval=interval, blit=True, repeat=repeat)
    return anim

def conv_result_anim(images, conv1, conv2, conv1_tf, conv2_tf, shape=[28,28], cmap='bone_r'):

    fig, ax = plt.subplots(1,3,figsize=(12,4))
    image1 = ax[0].imshow(images[0].reshape(shape[0],shape[1]), cmap=cmap)
    ax[0].set_title('Input Image')

    image2 = ax[1].imshow(conv1[0][:,:,np.random.randint(0,conv1_tf.shape[-1].value)].reshape(conv1_tf.shape[1].value,conv1_tf.shape[2].value), cmap=cmap)
    ax[1].set_title('Convolution Layer 1')

    image3 = ax[2].imshow(conv2[0][:,:,np.random.randint(0,conv2_tf.shape[-1].value)].reshape(conv2_tf.shape[1].value,conv2_tf.shape[2].value), cmap=cmap)

    ax[2].set_title('Convolution Layer 2')

    def update(i):
            image1 = ax[0].imshow(images[i].reshape(shape[0],shape[1]), cmap=cmap)

            image2 = ax[1].imshow(conv1[i][:,:,np.random.randint(0,conv1_tf.shape[-1].value)].reshape(conv1_tf.shape[1].value,conv1_tf.shape[2].value), cmap=cmap)

            image3 = ax[2].imshow(conv2[i][:,:,np.random.randint(0,conv2_tf.shape[-1].value)].reshape(conv2_tf.shape[1].value,conv2_tf.shape[2].value), cmap=cmap)

            return image1, image2, image3
    plt.tight_layout()
        
    return animation.FuncAnimation(fig, update, frames=len(images), interval=450, blit=True, repeat=True)
    
