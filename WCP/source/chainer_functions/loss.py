import chainer
import chainer.functions as F
from chainer import Variable, optimizers, cuda, serializers

def kl_binary(p_logit, q_logit):
    if isinstance(p_logit, chainer.Variable):
        xp = cuda.get_array_module(p_logit.data)
    else:
        xp = cuda.get_array_module(p_logit)
    p_logit = F.concat([p_logit, xp.zeros(p_logit.shape, xp.float32)], 1)
    q_logit = F.concat([q_logit, xp.zeros(q_logit.shape, xp.float32)], 1)
    return kl_categorical(p_logit, q_logit)


def kl_categorical(p_logit, q_logit):
    if isinstance(p_logit, chainer.Variable):
        xp = cuda.get_array_module(p_logit.data)
    else:
        xp = cuda.get_array_module(p_logit)
    p = F.softmax(p_logit)
    _kl = F.sum(p * (F.log_softmax(p_logit) - F.log_softmax(q_logit)), 1)
    return F.sum(_kl) / xp.prod(xp.array(_kl.shape))


def cross_entropy(logit, y):
    # y should be one-hot encoded probability
    return - F.sum(y * F.log_softmax(logit)) / logit.shape[0]


def kl(p_logit, q_logit):
    if p_logit.shape[1] == 1:
        return kl_binary(p_logit, q_logit)
    else:
        return kl_categorical(p_logit, q_logit)


def distance(p_logit, q_logit, dist_type="KL"):
    if dist_type == "KL":
        return kl(p_logit, q_logit)
    else:
        raise NotImplementedError


def entropy_y_x(p_logit):
    p = F.softmax(p_logit)
    return - F.sum(p * F.log_softmax(p_logit)) / p_logit.shape[0]

def get_normalized_vector(d, xp):
    d /= (1e-12 + xp.max(xp.abs(d), range(1, len(d.shape)), keepdims=True))
    d /= xp.sqrt(1e-6 + xp.sum(d ** 2, range(1, len(d.shape)), keepdims=True))
    return d
    
def delta_forward(forward, x, x_d, w_d, size_list):
    h = x + x_d
    h = F.convolution_2d(h, forward.c1.W + w_d[:size_list[0]].reshape(forward.c1.W.shape), stride=1, pad=1)
    h = F.leaky_relu(forward.bn1(h), slope=0.1)
    h = F.convolution_2d(h, forward.c2.W + w_d[size_list[0]:size_list[1]].reshape(forward.c2.W.shape), stride=1, pad=1)
    h = F.leaky_relu(forward.bn2(h), slope=0.1)
    h = F.convolution_2d(h, forward.c3.W + w_d[size_list[1]:size_list[2]].reshape(forward.c3.W.shape), stride=1, pad=1)
    h = F.leaky_relu(forward.bn3(h), slope=0.1)
    h = F.max_pooling_2d(h, ksize=2, stride=2)
    h = F.dropout(h, ratio=forward.dropout_rate)

    h = F.convolution_2d(h, forward.c4.W + w_d[size_list[2]:size_list[3]].reshape(forward.c4.W.shape), stride=1, pad=1)
    h = F.leaky_relu(forward.bn4(h), slope=0.1)
    h = F.convolution_2d(h, forward.c5.W + w_d[size_list[3]:size_list[4]].reshape(forward.c5.W.shape), stride=1, pad=1)
    h = F.leaky_relu(forward.bn5(h), slope=0.1)
    h = F.convolution_2d(h, forward.c6.W + w_d[size_list[4]:size_list[5]].reshape(forward.c6.W.shape), stride=1, pad=1)
    h = F.leaky_relu(forward.bn6(h), slope=0.1)
    h = F.max_pooling_2d(h, ksize=2, stride=2)
    h = F.dropout(h, ratio=forward.dropout_rate)

    h = F.convolution_2d(h, forward.c7.W + w_d[size_list[5]:size_list[6]].reshape(forward.c7.W.shape), stride=1, pad=0)
    h = F.leaky_relu(forward.bn7(h), slope=0.1)
    h = F.convolution_2d(h, forward.c8.W + w_d[size_list[6]:size_list[7]].reshape(forward.c8.W.shape), stride=1, pad=0)
    h = F.leaky_relu(forward.bn8(h), slope=0.1)
    h = F.convolution_2d(h, forward.c9.W + w_d[size_list[7]:size_list[8]].reshape(forward.c9.W.shape), stride=1, pad=0)
    h = F.leaky_relu(forward.bn9(h), slope=0.1)
    h = F.average_pooling_2d(h, ksize=h.data.shape[2])
    logit_perturb = F.linear(h, forward.l_cl.W + w_d[size_list[8]:].reshape(forward.l_cl.W.shape), forward.l_cl.b)
    if forward.top_bn:
        logit_perturb = forward.bn_cl(logit_perturb)
    
    return logit_perturb

    
def drop_forward(forward, x, drop_d_list):
    h = x
    h = F.convolution_2d(h, forward.c1.W, stride=1, pad=1)
    h = F.leaky_relu(forward.bn1(h), slope=0.1)
    h = F.convolution_2d(h, forward.c2.W * (1 - drop_d_list[0]).reshape(forward.c2.W.shape), stride=1, pad=1)
    h = F.leaky_relu(forward.bn2(h), slope=0.1)
    h = F.convolution_2d(h, forward.c3.W, stride=1, pad=1)
    h = F.leaky_relu(forward.bn3(h), slope=0.1)
    h = F.max_pooling_2d(h, ksize=2, stride=2)
    h = F.dropout(h, ratio=forward.dropout_rate)

    h = F.convolution_2d(h, forward.c4.W, stride=1, pad=1)
    h = F.leaky_relu(forward.bn4(h), slope=0.1)
    h = F.convolution_2d(h, forward.c5.W * (1 - drop_d_list[1]).reshape(forward.c5.W.shape), stride=1, pad=1)
    h = F.leaky_relu(forward.bn5(h), slope=0.1)
    h = F.convolution_2d(h, forward.c6.W, stride=1, pad=1)
    h = F.leaky_relu(forward.bn6(h), slope=0.1)
    h = F.max_pooling_2d(h, ksize=2, stride=2)
    h = F.dropout(h, ratio=forward.dropout_rate)

    h = F.convolution_2d(h, forward.c7.W, stride=1, pad=0)
    h = F.leaky_relu(forward.bn7(h), slope=0.1)
    h = F.convolution_2d(h, forward.c8.W * (1 - drop_d_list[2]).reshape(forward.c8.W.shape), stride=1, pad=0)
    h = F.leaky_relu(forward.bn8(h), slope=0.1)
    h = F.convolution_2d(h, forward.c9.W, stride=1, pad=0)
    h = F.leaky_relu(forward.bn9(h), slope=0.1)
    h = F.average_pooling_2d(h, ksize=h.data.shape[2])
    logit_perturb = F.linear(h, forward.l_cl.W, forward.l_cl.b)
    if forward.top_bn:
        logit_perturb = forward.bn_cl(logit_perturb)
    
    return logit_perturb
   
def wcp_loss(forward, x, logit, epsilon=8., dr=0.5, num_simulations=1, xi=1e-6):
    xp = cuda.get_array_module(x)
    n_batch = x.shape[0]
    
    size_list = []
    size_sum = 0
    for ii in range(10):
        size = forward.layer_list[ii].W.size
        size_sum += size
        size_list += [size_sum]
        
    d = xp.random.normal(size=x.shape)
    d /= (1e-12 + xp.max(xp.abs(d),range(1, len(d.shape)), keepdims=True))
    d /= xp.sqrt(1e-6 + xp.sum(d ** 2, range(1, len(d.shape)), keepdims=True))
    
    d_weight = xp.random.normal(size=size_list[-1])
    d_weight /= (1e-12 + xp.max(xp.abs(d_weight)))
    d_weight /= xp.sqrt(1e-6 + xp.sum(d_weight ** 2))
    
    drop_weight_list = []
    for ii in range(3):
        drop_weight = xp.random.normal(size=forward.layer_list[ii*3+1].W.size + 1)
        drop_weight /= (1e-12 + xp.max(xp.abs(drop_weight)))
        drop_weight /= xp.sqrt(1e-6 + xp.sum(drop_weight ** 2))
        drop_weight_list += [drop_weight]
    
    for _ in range(num_simulations):
        x_d = Variable(xi * d.astype(xp.float32))
        w_d = Variable(xi * d_weight.astype(xp.float32))
        drop_d1_list = []
        drop_d2_list = []
        for ii in range(3):
            drop_d1 = Variable(xi * (drop_weight_list[ii][:-1] + drop_weight_list[ii][-1]).astype(xp.float32))
            drop_d2 = Variable(xi * drop_weight_list[ii][:-1].astype(xp.float32))
            drop_d1_list += [drop_d1]
            drop_d2_list += [drop_d2]   
            
        logit_d = delta_forward(forward, x, x_d, w_d, size_list)
        logit_drop1 = drop_forward(forward, x, drop_d1_list)
        logit_drop2 = drop_forward(forward, x, drop_d2_list)       

        kl_loss = distance(logit.data, logit_d)
        kl_loss_drop1 = distance(logit.data, logit_drop1)
        kl_loss_drop2 = distance(logit.data, logit_drop2)
        
        d, d_weight = chainer.grad([kl_loss], [x_d, w_d], enable_double_backprop=False)
        d = d / F.sqrt(F.sum(d ** 2, tuple(range(1, len(d.shape))), keepdims=True))
        d_weight = d_weight / F.sqrt(F.sum(d_weight ** 2))
        
        layer1_drop1, layer4_drop1, layer7_drop1 = chainer.grad([kl_loss_drop1], drop_d1_list, enable_double_backprop=False)
        layer1_drop2, layer4_drop2, layer7_drop2 = chainer.grad([kl_loss_drop2], drop_d2_list, enable_double_backprop=False)
        
        layer1_drop2 = F.reshape(F.sum(layer1_drop2), (1, 1))
        layer4_drop2 = F.reshape(F.sum(layer4_drop2), (1, 1))
        layer7_drop2 = F.reshape(F.sum(layer7_drop2), (1, 1))
        drop_weight_list[0] = F.flatten(F.concat([F.reshape(layer1_drop1, (-1, 1)), layer1_drop2], axis=0))
        drop_weight_list[1] = F.flatten(F.concat([F.reshape(layer4_drop1, (-1, 1)), layer4_drop2], axis=0))
        drop_weight_list[2] = F.flatten(F.concat([F.reshape(layer7_drop1, (-1, 1)), layer7_drop2], axis=0))
        drop_weight_list = [drop_weight_list[ii] / F.sqrt(F.sum(drop_weight_list[ii] ** 2)) for ii in range(3)]
    
    drop_mask_list = []
    for ii in range(3):
        if drop_weight_list[ii][-1].data < 0:
            drop_weight_list[ii] = -drop_weight_list[ii]
        
        rank = xp.argsort(drop_weight_list[ii][:-1].data)
        mask = rank < (1-dr) * float(drop_weight_list[ii][:-1].shape[0])
        drop_mask_list += [mask.astype('float32')]
        
    d = epsilon * d
    d_weight = epsilon * d_weight
    
    h = x + d
    h = F.convolution_2d(h, forward.c1.W + d_weight[:size_list[0]].reshape(forward.c1.W.shape), stride=1, pad=1)
    h = F.leaky_relu(forward.bn1(h), slope=0.1)
    h = F.convolution_2d(h, (forward.c2.W + d_weight[size_list[0]:size_list[1]].reshape(forward.c2.W.shape)) * Variable(drop_mask_list[0]).reshape(forward.c2.W.shape), stride=1, pad=1)
    h = F.leaky_relu(forward.bn2(h), slope=0.1)
    h = F.convolution_2d(h, forward.c3.W + d_weight[size_list[1]:size_list[2]].reshape(forward.c3.W.shape), stride=1, pad=1)
    h = F.leaky_relu(forward.bn3(h), slope=0.1)
    h = F.max_pooling_2d(h, ksize=2, stride=2)
    h = F.dropout(h, ratio=forward.dropout_rate)

    h = F.convolution_2d(h, forward.c4.W + d_weight[size_list[2]:size_list[3]].reshape(forward.c4.W.shape), stride=1, pad=1)
    h = F.leaky_relu(forward.bn4(h), slope=0.1)
    h = F.convolution_2d(h, (forward.c5.W + d_weight[size_list[3]:size_list[4]].reshape(forward.c5.W.shape)) * Variable(drop_mask_list[1]).reshape(forward.c5.W.shape), stride=1, pad=1)
    h = F.leaky_relu(forward.bn5(h), slope=0.1)
    h = F.convolution_2d(h, forward.c6.W + d_weight[size_list[4]:size_list[5]].reshape(forward.c6.W.shape), stride=1, pad=1)
    h = F.leaky_relu(forward.bn6(h), slope=0.1)
    h = F.max_pooling_2d(h, ksize=2, stride=2)
    h = F.dropout(h, ratio=forward.dropout_rate)

    h = F.convolution_2d(h, forward.c7.W + d_weight[size_list[5]:size_list[6]].reshape(forward.c7.W.shape), stride=1, pad=0)
    h = F.leaky_relu(forward.bn7(h), slope=0.1)
    h = F.convolution_2d(h, (forward.c8.W + d_weight[size_list[6]:size_list[7]].reshape(forward.c8.W.shape)) * Variable(drop_mask_list[2]).reshape(forward.c8.W.shape), stride=1, pad=0)
    h = F.leaky_relu(forward.bn8(h), slope=0.1)
    h = F.convolution_2d(h, forward.c9.W + d_weight[size_list[7]:size_list[8]].reshape(forward.c9.W.shape), stride=1, pad=0)
    h = F.leaky_relu(forward.bn9(h), slope=0.1)
    h = F.average_pooling_2d(h, ksize=h.data.shape[2])
    logit_perturb = F.linear(h, forward.l_cl.W + 0.1 * d_weight[size_list[8]:].reshape(forward.l_cl.W.shape), forward.l_cl.b)
    if forward.top_bn:
        logit_perturb = forward.bn_cl(logit_perturb)
        
    loss = distance(logit.data, logit_perturb)
    
    return loss