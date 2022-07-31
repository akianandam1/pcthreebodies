from ModelVisuals import optimize
import torch
from nhd import data


# Perturbs input vector using normal distribution
# takes in float standard deviation
# Requires floats
def perturb(vec, std):
    return torch.tensor([torch.normal(mean=vec[0], std=torch.tensor(std)),
                         torch.normal(mean=vec[1], std=torch.tensor(std)),
                         0.0,
                         torch.normal(mean=vec[3], std=torch.tensor(std)),
                         torch.normal(mean=vec[4], std=torch.tensor(std)),
                         0.0,
                         torch.normal(mean=vec[6], std=torch.tensor(std)),
                         torch.normal(mean=vec[7], std=torch.tensor(std)),
                         0.0,
                         torch.normal(mean=vec[9], std=torch.tensor(std)),
                         torch.normal(mean=vec[10], std=torch.tensor(std)),
                         0.0,
                         torch.normal(mean=vec[12], std=torch.tensor(std)),
                         torch.normal(mean=vec[13], std=torch.tensor(std)),
                         0.0,
                         torch.normal(mean=vec[15], std=torch.tensor(std)),
                         torch.normal(mean=vec[16], std=torch.tensor(std)),
                         0.0,], requires_grad = True)



# beginning tensor([-1.0018,  0.0289,  0.0000,  0.9649,  0.0147,  0.0000,  0.0129,  0.0171,
#          0.0000,  0.4700,  0.2530,  0.0000,  0.4089,  0.2995,  0.0000, -1.1218,
#         -0.6996,  0.0000,  1.0000,  1.0000,  0.7500], grad_fn=<CatBackward0>)

for input_set in data:
    m_1 = float(input_set[0])
    m_2 = float(input_set[1])
    m_3 = float(input_set[2])
    x_1 = float(input_set[3])
    v_1 = float(input_set[4])
    v_2 = float(input_set[5])
    T = float(input_set[6])
    vec = torch.tensor([x_1,0,0,1,0,0,0,0,0, 0, v_1, 0, 0, v_2, 0, 0, -(m_1*v_1 + m_2*v_2)/m_3, 0], requires_grad = True)
    print(vec)
    #vec = perturb(vec, .02)
    optimize(vec, m_1, m_2, m_3, lr = .0001, num_epochs = 90, max_period=int(T+2), video_folder = "NHD")

# m_1 = 1
# m_2 = 1
# m_3 = 0.75
#
# v_1 = 0.2827020949
# v_2 = 0.3272089716
# T = 10.9633031497
# vec = torch.tensor([-1,0,0,1,0,0,0,0,0,v_1, v_2, 0, v_1, v_2, 0, -2*v_1/m_3, -2*v_2/m_3, 0], requires_grad = True)
# vec = torch.tensor([-1,0,0, 1,0,0, 0,0,0, v_1,v_2,0, v_1,v_2,0, -2*v_1,-2*v_2,0], requires_grad=True)

# FIGURE 1 vec = torch.tensor([-1.0018,  0.0289,  0.0000,  0.9649,  0.0147,  0.0000,  0.0129,  0.0171,
#          0.0000,  0.4700,  0.2530,  0.0000,  0.4089,  0.2995,  0.0000, -1.1218,
#         -0.6996,  0.0000], requires_grad = True)


# vec = torch.tensor([-0.9937,  0.0193,  0.0000,  0.9634,  0.0237,  0.0000,  0.0061,  0.0221,
#          0.0000,  0.4581,  0.2431,  0.0000,  0.4085,  0.2965,  0.0000, -1.1173,
#         -0.6903,  0.0000], requires_grad = True)

# vec = perturb(vec, .01)
#optimize(vec, m_1, m_2, m_3, lr = .0001, num_epochs = 40, max_period=int(T+2), opt_func=torch.optim.Adagrad)
# optimize(vec, m_1, m_2, m_3, lr = .00001, num_epochs = 90, max_period=int(T+2), opt_func=torch.optim.SGD)

# vec = torch.tensor([-0.9957,  0.0225,  0.0000,  0.9649,  0.0178,  0.0000,  0.0070,  0.0205,
#          0.0000,  0.4640,  0.2438,  0.0000,  0.4100,  0.2989,  0.0000, -1.1197,
#         -0.6929,  0.0000], requires_grad = True)
#
#
# vec = vec = perturb(vec, .02)


