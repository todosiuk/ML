import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# load textfile dataset (2D data points)
# for each user, how many packets are sent per second and what's the size of a packet
# anomalies (DDOS attempts) will have lots of big packets sent in a short amount of time
def load_dataset(name):
    return np.loadtxt(name)


# euclidian distance between 2 data points. For as many data points as necessary.
def euclidian(a, b):
    return np.linalg.norm(a - b)

#lets define a plotting algorithm for our dataset and our centroids
def plot(dataset, history_centroids, belongs_to):
    # we'll have 2 colors for each centroid cluster
    colors = ['r', 'g']

    # split our graph by its axis and actual plot
    fig, ax = plt.subplots()

    # for each point in our dataset
    for index in range(dataset.shape[0]):
        # get all the points assigned to a cluster
        instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
        # assign each datapoint in that cluster a color and plot it
        for instance_index in instances_close:
            ax.plot(dataset[instance_index][0], dataset[instance_index][1], (colors[index] + 'o'))

    # lets also log the history of centroids calculated via training
    history_points = []
    # for each centroid ever calculated
    for index, centroids in enumerate(history_centroids):
        # print them all out
        for inner, item in enumerate(centroids):
            if index == 0:
                history_points.append(ax.plot(item[0], item[1], 'bo')[0])
            else:
                history_points[inner].set_data(item[0], item[1])
                print("centroids {} {}".format(index, item))

                plt.pause(0.8)


def kmeans(k, epsilon=0, distance='euclidian'):
    # list to store past centroid
    history_centroids = []
    # set the distance calculation type
    if distance == 'euclidian':
        dist_method = euclidian
    dataset = load_dataset('durudataset.txt')
    # dataset = dataset[:, 0:dataset.shape[1] - 1]
    # get the number of rows (instances) and columns (features) from the dataset
    num_instances, num_features = dataset.shape
    # define k centroids (how many clusters do we want to find?) chosen randomly
    prototypes = dataset[np.random.randint(0, num_instances - 1, size=k)]
    # set these to our list of past centroid (to show progress over time)
    history_centroids.append(prototypes)
    # to keep track of centroid at every iteration
    prototypes_old = np.zeros(prototypes.shape)
    # to store clusters
    belongs_to = np.zeros((num_instances, 1))
    norm = dist_method(prototypes, prototypes_old)
    iteration = 0
    while norm > epsilon:
        iteration += 1
        norm = dist_method(prototypes, prototypes_old)
        prototypes_old = prototypes
        # for each instance in the dataset
        for index_instance, instance in enumerate(dataset):
            # define a distance vector of size k
            dist_vec = np.zeros((k, 1))
            # for each centroid
            for index_prototype, prototype in enumerate(prototypes):
                # compute the distance between x and centroid
                dist_vec[index_prototype] = dist_method(prototype,
                                                        instance)
            # find the smallest distance, assign that distance to a cluster
            belongs_to[index_instance, 0] = np.argmin(dist_vec)

        tmp_prototypes = np.zeros((k, num_features))

        # for each cluster (k of them)
        for index in range(len(prototypes)):
            # get all the points assigned to a cluster
            instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
            # find the mean of those points, this is our new centroid
            prototype = np.mean(dataset[instances_close], axis=0)
            # prototype = dataset[np.random.randint(0, num_instances, size=1)[0]]
            # add our new centroid to our new temporary list
            tmp_prototypes[index, :] = prototype

        # set the new list to the current list
        prototypes = tmp_prototypes

        # add our calculated centroids to our history for plotting
        history_centroids.append(tmp_prototypes)

    # plot(dataset, history_centroids, belongs_to)

    # return calculated centroids, history of them all, and assignments for which cluster each datapoint belongs to
    return prototypes, history_centroids, belongs_to

#main file
def execute():
    # load dataset
    dataset = load_dataset('C:\\Users\\В\\Desktop\\OrganiZen\\Al-in-One 09-09-2018\\ML\\durudataset.txt')
    # train the model on the data
    centroids, history_centroids, belongs_to = kmeans(2)
    # plot the results
    plot(dataset, history_centroids, belongs_to)


execute()
