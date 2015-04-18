import sys
import random
import numpy
from pyspark import SparkContext, SparkConf


# Unused (for parsing Netflix raw data)
# Returns a triplet of (movieid, userid, rating) for each line in the files
def file2triplets(infile):
    lines = infile[1].split("\n")
    return [(int(lines[0].strip(':')), int(line.split(",")[0]), int(line.split(",")[1])) for line in lines[1:]]

# Hash function returning block idx given userid or movieid
def hashfunc(idx, numworkers, seed):
    return hash(str(idx) + str(seed)) % numworkers

# Function to update W and H
def updateWH((Vblock, Wblock, Hblock), num_updates, beta_val, lambda_val, Ni, Nj):
    Wdict = dict(Wblock)
    Hdict = dict(Hblock)
    it=0
    for (movieid, userid, rating) in Vblock:
        # Compute the number of updates
        it += 1
        eps_val = pow(100+num_updates+it, -beta_val)  
        Wi = Wdict[movieid]
        Hj = Hdict[userid]
        WiHj = numpy.dot(Wi,Hj)
        # L_NZSL loss gradient coefficient
        LNZSL_coeff = -2*(rating - WiHj)
        Wdict[movieid] = Wi - eps_val*(LNZSL_coeff*Hj + 2*lambda_val/Ni[movieid]*Wi)      
        Hdict[userid]  = Hj - eps_val*(LNZSL_coeff*Wi + 2*lambda_val/Nj[userid]*Hj)
    return (Wdict.items(), Hdict.items())

# Loss function
def lossNZSL(Ventry, W, H):
    return pow(Ventry[2] - numpy.dot(W[Ventry[0]],H[Ventry[1]]),2)

# Read command line arguments
num_factors = int(sys.argv[1])
num_workers = int(sys.argv[2])
num_iterations = int(sys.argv[3])
beta_val = float(sys.argv[4])
lambda_val = float(sys.argv[5])
inputV_path = sys.argv[6]
outputW_path = sys.argv[7]
outputH_path = sys.argv[8]

conf = SparkConf().setAppName("dsgd_mf")
sc = SparkContext(conf=conf)

# To parse raw netflix data
# V = sc.wholeTextFiles(inputV_path)
# triples = V.flatMap(file2triplets)

triples = sc.textFile(inputV_path).map(lambda a: [int(x) for x in a.split(",")])

triples.persist()
num_movies = triples.map(lambda trip : trip[0]).reduce(max)
num_users = triples.map(lambda trip : trip[1]).reduce(max)
Ni = triples.keyBy(lambda trip: trip[0]).countByKey()
Nj = triples.keyBy(lambda trip: trip[1]).countByKey()

# W*H = V
# W is a list of (movieid, factors) kv pairs 
# H is a list of (userid, factors) kv pairs
#   where factors is a list of floats of length num_factors
W = sc.parallelize(range(num_movies+1)).map(lambda a : (a, 0.5*numpy.random.rand(num_factors))).persist()
H = sc.parallelize(range(num_users+1)).map(lambda a : (a, 0.5*numpy.random.rand(num_factors))).persist()

num_updates = 0
loss_all = []

for it in range(num_iterations):
    seed = random.randrange(100000)
    # Get the diagonal blocks of V
    filtered = triples.filter(lambda trip : hashfunc(trip[0],num_workers,seed) == hashfunc(trip[1],num_workers,seed)).persist()
    Vblocks = filtered.keyBy(lambda trip : hashfunc(trip[0], num_workers, seed))
    cur_num_updates = filtered.count()
    filtered.unpersist()    
    # Partition W and H and group them with V by they block number
    Wblocks = W.keyBy(lambda pair: hashfunc(pair[0], num_workers, seed))
    Hblocks = H.keyBy(lambda pair: hashfunc(pair[0], num_workers, seed))
    grouped = Vblocks.groupWith(Wblocks, Hblocks).coalesce(num_workers)
    # Perform the updates to W and H in parallel
    updatedWH = grouped.map(lambda a: updateWH(a[1], num_updates, beta_val, lambda_val, Ni, Nj)).persist()
    W = updatedWH.flatMap(lambda a: a[0]).persist()
    H = updatedWH.flatMap(lambda a: a[1]).persist()
    num_updates += cur_num_updates


Wpy = numpy.vstack(W.sortByKey().map(lambda a : a[1]).collect()[1:])
numpy.savetxt(outputW_path, Wpy, delimiter=',')

Hpy = numpy.vstack(H.sortByKey().map(lambda a : a[1]).collect()[1:])
numpy.savetxt(outputH_path, Hpy.T, delimiter=',')


