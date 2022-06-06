import numpy as np
import scipy.spatial.distance as distance


def CURE(data, num_representative_points, alpha, num_cluster):
    numPts = len(data)
    Clusters = append_cluster(data)
    distCluster = fill_distCluster(data, numPts, Clusters)
    Clusters = update_distCluster(len(data), num_cluster, Clusters, distCluster, num_representative_points, alpha)
    Y_pred = [0] * numPts
    for i in range(0, len(Clusters)):
        for j in range(0, len(Clusters[i].index)):
            Y_pred[Clusters[i].index[j]] = i + 1
    return Y_pred


def dist(vecA, vecB):
    return np.sqrt(np.power(vecA - vecB, 2).sum())


def append_cluster(data):
    Clusters = []
    for idPoint in range(len(data)):
        newClust = CureCluster(idPoint, data[idPoint, :])
        Clusters.append(newClust)
    return Clusters


def fill_distCluster(data, numPts, Clusters):
    distCluster = (np.ones([len(data), len(data)])) * float('inf')
    for row in range(0, numPts):
        for col in range(0, row):
            distCluster[row][col] = dist(Clusters[row].center, Clusters[col].center)
    return distCluster


def update_distCluster(numCluster, num_cluster, Clusters, distCluster, num_representative_points, alpha):
    while numCluster > num_cluster:
        if np.mod(numCluster, 50) == 0:
            print('Cluster count:', numCluster)
        # Find a pair of closet clusters
        minIndex = np.where(distCluster == np.min(distCluster))
        minIndex1 = minIndex[0][0]
        minIndex2 = minIndex[1][0]
        # Merge
        Clusters[minIndex1].mergeWithCluster(Clusters[minIndex2], num_representative_points, alpha)
        # Update the distCluster matrix
        for i in range(0, minIndex1):
            distCluster[minIndex1, i] = Clusters[minIndex1].distRep(Clusters[i])
        for i in range(minIndex1 + 1, numCluster):
            distCluster[i, minIndex1] = Clusters[minIndex1].distRep(Clusters[i])
        # Delete the merged cluster and its disCluster vector.
        distCluster = np.delete(distCluster, minIndex2, axis=0)
        distCluster = np.delete(distCluster, minIndex2, axis=1)
        del Clusters[minIndex2]
        numCluster = numCluster - 1
    return Clusters


# CURE CLUSTERING MODEL
class CureCluster:
    def __init__(self, id__, center__):
        self.points = center__
        self.repPoints = center__
        self.center = center__
        self.index = [id__]

    # Computes and stores the centroid of this cluster, based on its points
    def computeCentroid(self, clust):
        totalPoints_1 = len(self.index)
        totalPoints_2 = len(clust.index)
        self.center = (self.center * totalPoints_1 + clust.center * totalPoints_2) / (totalPoints_1 + totalPoints_2)

    # Computes and stores representative points for this cluster
    def generateRepPoints(self, numRepPoints, alpha):
        tempSet = None
        for i in range(1, numRepPoints + 1):
            maxDist = 0
            maxPoint = None
            for p in range(0, len(self.index)):
                if i == 1:
                    minDist = dist(self.points[p, :], self.center)
                else:
                    X = np.vstack([tempSet, self.points[p, :]])
                    tmpDist = distance.pdist(X)
                    minDist = tmpDist.min()
                if minDist >= maxDist:
                    maxDist = minDist
                    maxPoint = self.points[p, :]
            if tempSet is None:
                tempSet = maxPoint
            else:
                tempSet = np.vstack((tempSet, maxPoint))
        for j in range(len(tempSet)):
            if self.repPoints is None:
                self.repPoints = tempSet[j, :] + alpha * (self.center - tempSet[j, :])
            else:
                self.repPoints = np.vstack((self.repPoints, tempSet[j, :] + alpha * (self.center - tempSet[j, :])))

    # Computes and stores distance between this cluster and the other one.
    def distRep(self, clust):
        distRep = float('inf')
        for repA in self.repPoints:
            if type(clust.repPoints[0]) != list:
                repB = clust.repPoints
                distTemp = dist(repA, repB)
                if distTemp < distRep:
                    distRep = distTemp
            else:
                for repB in clust.repPoints:
                    distTemp = dist(repA, repB)
                    if distTemp < distRep:
                        distRep = distTemp
        return distRep

    # Merges this cluster with the given cluster, recomputing the centroid and the representative points.
    def mergeWithCluster(self, clust, numRepPoints, alpha):
        self.computeCentroid(clust)
        self.points = np.vstack((self.points, clust.points))
        self.index = np.append(self.index, clust.index)
        self.repPoints = None
        self.generateRepPoints(numRepPoints, alpha)
