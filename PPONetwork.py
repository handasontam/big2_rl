import tensorflow.compat.v1 as tf
import numpy as np
# from baselines.a2c.utils import fc
import joblib
tf.disable_v2_behavior()


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b

class PPONetwork(object):
    
    def __init__(self, sess, obs_dim, act_dim, name):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.name = name
        
        with tf.variable_scope(name):
            X = tf.placeholder(tf.float32, [None, obs_dim], name="input")
            available_moves = tf.placeholder(tf.float32, [None, act_dim], name="availableActions")
            #available_moves takes form [0, 0, -inf, 0, -inf...], 0 if action is available, -inf if not.
            activation = tf.nn.relu
            h1_shared = activation(fc(X,'fc1', nh=512, init_scale=np.sqrt(2)))
            h2_shared = activation(fc(h1_shared,'fc1_shared', nh=512, init_scale=np.sqrt(2))) + h1_shared  # Residual
            h3_shared = activation(fc(h2_shared,'fc2_shared', nh=512, init_scale=np.sqrt(2))) + h2_shared  # Residual
            h4_shared = activation(fc(h3_shared,'fc3_shared', nh=512, init_scale=np.sqrt(2))) + h3_shared  # Residual
            h5_shared = activation(fc(h4_shared,'fc4_shared', nh=512, init_scale=np.sqrt(2))) + h4_shared  # Residual
            pi_head = activation(fc(h5_shared,'fc2', nh=512, init_scale=np.sqrt(2)))
            pi = fc(pi_head,'pi', act_dim, init_scale = 0.5)
            #value function - share layer h2_shared
            value_haed = activation(fc(h5_shared,'fc3', nh=512, init_scale=0.5))
            vf = fc(value_haed, 'vf', 1)[:,0]
        availPi = tf.add(pi, available_moves)    
        
        def sample():
            u = tf.random.uniform(tf.shape(availPi))
            return tf.argmax(availPi - tf.math.log(-tf.math.log(u)), axis=-1)
        
        a0 = sample()
        p0in = tf.nn.softmax(availPi)
        onehot = tf.one_hot(a0, availPi.get_shape().as_list()[-1])
        neglogpac = -tf.log(tf.reduce_sum(tf.multiply(p0in, onehot), axis=-1))
        
        def step(obs, availAcs):
            a, v, neglogp = sess.run([a0, vf, neglogpac], {X:obs, available_moves:availAcs})
            return a, v, neglogp
            
        def value(obs, availAcs):
            return sess.run(vf, {X:obs, available_moves:availAcs})
        
        self.X = X
        self.availPi = availPi
        self.available_moves = available_moves
        self.pi = pi
        self.vf = vf        
        self.step = step
        self.value = value
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        
        def getParams():
            return sess.run(self.params)
        
        self.getParams = getParams
        
        def loadParams(paramsToLoad):
            restores = []
            for p, loadedP in zip(self.params, paramsToLoad):
                restores.append(p.assign(loadedP))
            sess.run(restores)
            
        self.loadParams = loadParams
        
        def saveParams(path):
            modelParams = sess.run(self.params)
            joblib.dump(modelParams, path)
            
        self.saveParams = saveParams
     
        
        
class PPOModel(object):
    
    def __init__(self, sess, network, inpDim, actDim, ent_coef, vf_coef, max_grad_norm, l2_coef):
        
        self.network = network
        
        #placeholder variables
        ACTIONS = tf.placeholder(tf.int32, [None], name='actionsPlaceholder')
        ADVANTAGES = tf.placeholder(tf.float32, [None], name='advantagesPlaceholder')
        RETURNS = tf.placeholder(tf.float32, [None], name='returnsPlaceholder')
        OLD_NEG_LOG_PROB_ACTIONS = tf.placeholder(tf.float32,[None], name='oldNegLogProbActionsPlaceholder')
        OLD_VAL_PRED = tf.placeholder(tf.float32,[None], name='oldValPlaceholder')
        LEARNING_RATE = tf.placeholder(tf.float32,[], name='LRplaceholder')
        CLIP_RANGE = tf.placeholder(tf.float32,[], name='cliprangePlaceholder')
        VF_CLIP_RANGE = tf.placeholder(tf.float32,[], name='vfcliprangePlaceholder')
        
        p0 = tf.nn.softmax(network.availPi)
        entropy = -tf.reduce_sum((p0+1e-8) * tf.log(p0+1e-8), axis=-1)
        oneHotActions = tf.one_hot(ACTIONS, network.pi.get_shape().as_list()[-1])
        neglogpac = -tf.log(tf.reduce_sum(tf.multiply(p0, oneHotActions), axis=-1))
        
        def neglogp(state, actions, index):
            return sess.run(neglogpac, {network.X: state, network.available_moves: actions, ACTIONS: index})
        
        self.neglogp = neglogp
        
        #define loss functions
        #entropy loss
        entropyLoss = tf.reduce_mean(entropy)
        #value loss
        v_pred = network.vf
        v_pred_clipped = OLD_VAL_PRED + tf.clip_by_value(v_pred - OLD_VAL_PRED, -VF_CLIP_RANGE, VF_CLIP_RANGE)
        vf_losses1 = tf.square(v_pred - RETURNS)
        vf_losses2 = tf.square(v_pred_clipped - RETURNS)
        vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        #policy gradient loss
        prob_ratio = tf.exp(OLD_NEG_LOG_PROB_ACTIONS - neglogpac)
        pg_losses1 = -ADVANTAGES * prob_ratio
        pg_losses2 = -ADVANTAGES * tf.clip_by_value(prob_ratio, 1.0-CLIP_RANGE, 1.0+CLIP_RANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))
        # l2 regularization
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in network.params if 'b' not in v.name])
        # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in network.params])
        #total loss
        loss = pg_loss + vf_coef*vf_loss - ent_coef*entropyLoss + l2_coef*l2_loss
        
        params = network.params
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.0, epsilon=1e-7)
        _train = trainer.apply_gradients(grads)
        
        def train(lr, cliprange, vf_cliprange, observations, availableActions, returns, actions, values, neglogpacs):
            advs = returns - values
            advs = (advs-advs.mean()) / (advs.std() + 1e-8)
            inputMap = {network.X: observations, network.available_moves: availableActions, ACTIONS: actions, ADVANTAGES: advs, RETURNS: returns,
                        OLD_VAL_PRED: values, OLD_NEG_LOG_PROB_ACTIONS: neglogpacs, LEARNING_RATE: lr, CLIP_RANGE: cliprange, VF_CLIP_RANGE: vf_cliprange}
            return sess.run([pg_loss, vf_loss, entropyLoss, _train], inputMap)[:-1]
        
        self.train = train
