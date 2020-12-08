_border_widths = [6, 8, 10, 12]
_border_heights = [6, 8, 10, 12]
default_width = _border_widths[0]
default_height = _border_heights[0]
channel = 5
players = {0: 'black', 1: 'white'}

# train params
# 正则化参数c  loss = (z-v)**2 + πlogp + c||w||**2
# paper: L2 regularisation parameter is set to c = 10 4
l2_c = 1e-4
lr = 1e-4
batch_size = 512
buffer_size = 100000
# where cpuct is a constant determining the level of exploration; this search control strategy initially
# prefers actions with high prior probability and low visit count, but asympotically prefers actions
# with high action-value.
c_put = 5
t = 1
# Over the course of training, 4.9 million games of self-play were generated,
# using 1,600 simulations for each MCTS, which corresponds to approximately 0.4s thinking time per move.
# use 8 * 8  divide 5
ratio = (19 * 19) // (default_width * default_height)
per_search_simulation_num = 1600 // ratio

# save weigths path
save_path = '../model/'
record_path = '../model/record.pkl'