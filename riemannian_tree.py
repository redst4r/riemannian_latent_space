import numpy as np
import tensorflow as tf
import tqdm

"""
along the lines of 

Fast Approximate Geodesics for Deep Generative Models
Nutan Chen, Francesco Ferroni, Alexej Klushyn, Alexandros Paraschos, Justin Bayer, Patrick van der Smagt

"""

class RiemannianMetric(object):
    def __init__(self, x, z, session):
        self.x = x
        self.z = z
        self.session = session

    def create_tf_graph(self):
        """
        creates the metric tensor (J^T J and J being the jacobian of the decoder), 
        which can be evaluated at any point in Z
        and
        the magnification factor
        """

        # the metric tensor
        output_dim = self.x.shape[1].value
        # derivative of each output dim wrt to input (tf.gradients would sum over the output)
        J = [tf.gradients(self.x[:, _], self.z)[0] for _ in range(output_dim)]
        J = tf.stack(J, axis=1)  # batch x output x latent
        self.J = J

        G = tf.transpose(J, [0, 2, 1]) @ J  # J^T \cdot J
        self.G = G

        # magnification factor
        MF = tf.sqrt(tf.linalg.det(G))
        self.MF = MF

    def riemannian_distance_along_line(self, z1, z2, n_steps):
        """
        calculates the riemannian distance between two near points in latent space on a straight line
        the formula is L(z1, z2) = \int_0^1 dt \sqrt(\dot \gamma^T J^T J \dot gamma)
        since gamma is a straight line \gamma(t) = t z_1 + (1-t) z_2, we get
        L(z1, z2) = \int_0^1 dt \sqrt([z_1 - z2]^T J^T J [z1-z2])
        L(z1, z2) = \int_0^1 dt \sqrt([z_1 - z2]^T G [z1-z2])

        z1: starting point
        z2: end point
        n_steps: number of discretization steps of the integral
        """

        # discretize the integral aling the line
        t = np.linspace(0, 1, n_steps)
        dt = t[1] - t[0]
        the_line = np.concatenate([_ * z1 + (1 - _) * z2 for _ in t])

        if True:
            # for weird reasons it seems to be alot faster to first eval G then do matrix mutliple outside of TF
            G_eval = self.session.run(self.G, feed_dict={self.z: the_line})

            # eval the integral at discrete point
            L_discrete = np.sqrt((z1-z2) @ G_eval @ (z1-z2).T)
            L_discrete = L_discrete.flatten()

            L = np.sum(dt * L_discrete)

        else:
            # THIS IS ALOT (10x) slower, although its all in TF
            DZ = tf.constant(z1 - z2)
            DZT = tf.constant((z1 - z2).T)
            tmp_ = tf.tensordot(self.G, DZT, axes=1)
            tmp_ = tf.einsum('j,ijk->ik', DZ[0], tmp_ )
            # tmp_ = tf.tensordot(DZ, tmp_, axes=1)

            L_discrete = tf.sqrt(tmp_)  # this is a function of z, since G(z)

            L_eval = self.session.run(L_discrete, feed_dict={self.z: the_line})
            L_eval = L_eval.flatten()
            L = np.sum(dt * L_eval)

        return L


class RiemannianTree(object):
    """docstring for RiemannianTree"""
    def __init__(self, riemann_metric):
        super(RiemannianTree, self).__init__()
        self.riemann_metric = riemann_metric  # decoder input (tf_variable)

    def create_nn_graph(self, encoded_x, n_steps, n_neighbors):
        """
        nearst neightoburs in latent space
        (still based on euclidean distance in latent space!!)

        but puts the riemannian distance on the edges of nearest neightbours
        """
        from sklearn.neighbors import NearestNeighbors
        n_data = len(encoded_x)
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        knn.fit(encoded_x)

        # now, for each datapoint and nearest neightoburs,
        # calculate the Riemannian distance and the heuristic
        # distance for the A*-algorithm

        riemannian_matrix = np.full([n_data, n_data], np.inf)
        heuristic_matrix = np.full([n_data, n_data], np.inf)
        for i in tqdm.trange(n_data):
            z = encoded_x[i:i+1]

            distances, indices = knn.kneighbors(z)

            # first dim is for samples (z), but we only have one
            distances = distances[0]
            indices = indices[0]
            for ix, dist in zip(indices, distances):
                # calculate the riemannian distance of z and its nn

                # save some computation if we alrdy calculated the other direction
                if riemannian_matrix[i, ix] != np.inf or riemannian_matrix[ix, i] != np.inf:
                    continue

                L_riemann = self.riemann_metric.riemannian_distance_along_line(z, encoded_x[ix], n_steps=n_steps)
                L_heuristic = dist

                riemannian_matrix[i, ix] = L_riemann
                heuristic_matrix[i, ix] = L_heuristic

                # Note that NN are not symmetric, but lets make the matrix symmetric
                riemannian_matrix[ix, i] = L_riemann
                heuristic_matrix[ix, i] = L_heuristic
        # TODO: not sure if the heutrisitc matrix should contain inf!!
        # maybe just euclidean distance of all-vs-all
        return riemannian_matrix, heuristic_matrix

    def create_nx_graph(self, encoded_x, n_steps, n_neighbors):
        A, B  = self.create_nn_graph(encoded_x, n_steps, n_neighbors)

        def _prep_network(A):
            # in networkx, the floats are edge weights, i.e. the larger the stronger connected
            # but my floats are distances
            weights = 1/A
            weights[weights==np.inf] = 0
            import networkx as nx
            G = nx.Graph(weights)

            for i in range(len(encoded_x)):
                G.nodes[i]['z1'] = float(encoded_x[i,0])
                G.nodes[i]['z2'] = float(encoded_x[i,1])
                G.nodes[i]['label'] = str(i)

            for n1,n2 in G.edges:
                G.edges[n1,n2]['distance'] = float(A[n1,n2])

            return G

        G_riemann = _prep_network(A)
        G_euclidean = _prep_network(B)

        return G_riemann, G_euclidean



def main():
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Input
    latent_dim = 2
    output_dim = 1
    m = Sequential()
    m.add(Dense(200, activation='tanh', input_shape=(latent_dim, )))
    m.add(Dense(200, activation='tanh', ))
    m.add(Dense(output_dim, activation='tanh'))


    # plot the model real quick
    inp = np.random.uniform(-50,50, size=(1000, latent_dim))
    outp = m.predict(inp)

    plt.figure()
    plt.scatter(inp[:,0], inp[:,1])
    plt.figure()
    plt.scatter(outp[:,0], outp[:,1])


    session = tf.Session()
    session.run(tf.global_variables_initializer())

    rmetric = RiemannianMetric(x=m.output, z=m.input, session=session)
    rmetric.create_tf_graph()

    mf = session.run(rmetric.MF, {rmetric.z: inp})
    plt.figure()
    plt.scatter(inp[:,0], inp[:,1], c=mf)

    z1 = np.array([[1, 10]])
    z2 = np.array([[10, 2]])

    # for steps in [100,1_000,10_000,100_000]:
    #     q = r.riemannian_distance_along_line(z1, z2, n_steps=steps)
    #     print(q)


    import sklearn.datasets
    z, _  = sklearn.datasets.make_swiss_roll(n_samples=1000, noise=0.5, random_state=None)
    z = z[:,[0,2]]

    z = np.random.uniform(-50,50, size=(1000, latent_dim))

    # plt.scatter(z[:,0], z[:,1])
    outp = m.predict(z)
    plt.figure()
    plt.scatter(outp[:,0], outp[:,1])

    rTree = RiemannianTree(rmetric)

    A, B = rTree.create_nn_graph(z, n_steps=1000, n_neighbors=10)

    #ahoertest path finding
    # for nx a non-existing edge should be 0 weight
    A = 1/A
    A[A==np.inf] = 0
    import networkx as nx
    G = nx.Graph(A)


if __name__ == '__main__':
    main()


