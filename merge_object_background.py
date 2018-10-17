# Create mask with black foreground meaning the object to be placed.
image_4channel = cv2.imread('mexican_puro.png', cv2.IMREAD_UNCHANGED)
height, width = image_4channel.shape[:2]

# Background image.
bck = cv2.imread('mexican_transformation_2.png', cv2.IMREAD_UNCHANGED)
bck = cv2.cvtColor(bck, cv2.COLOR_BGR2RGB)
bck = cv2.resize(bck, (width, height))

# Create mask that allows pixels of the object pass.
#image_4channel = cv2.imread('mexican_bigote.png', cv2.IMREAD_UNCHANGED)
height, width = image_4channel.shape[:2]
alpha = image_4channel[:, :, 3].copy()
alphaf = alpha.flatten()
# Makes black the background.
# alphaf[np.where(alphaf==0)[0]] = 0
# alphaf[np.where(alphaf>=1)[0]] = 1
# Makes black the region where the puro should be.
alphaf[np.where(alphaf==0)[0]] = 1
alphaf[np.where(alphaf!=1)[0]] = 0
a = alphaf.reshape(height, width)
empty = np.zeros([height, width, 3], np.uint8)
empty[:, :, 0] = a
empty[:, :, 1] = a
empty[:, :, 2] = a

bck = empty*bck
# plot_image(bck)

# Create object with black background.
# image_4channel = cv2.imread('mexican_puro.png', cv2.IMREAD_UNCHANGED)
alpha_channel = image_4channel[:,:,3]
rgb_channels = image_4channel[:,:,:3]

# White Background Image
white_background_image = np.zeros_like(rgb_channels, dtype=np.uint8)

# Alpha factor
alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) #/ 255.0
alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)

# Transparent Image Rendered on White Background
base = rgb_channels.astype(np.float32) * alpha_factor
white = white_background_image.astype(np.float32) * (1 - alpha_factor)
final_image = base + white
final_image = final_image.astype(np.uint8)
# plot_image(final_image)

# In order to merge the object and the bacground.
new = cv2.add(final_image, bck)
new = cv2.cvtColor(new, cv2.COLOR_BGR2RGB)
cv2.imwrite('new_image.png', new)
#plot_image(new)
