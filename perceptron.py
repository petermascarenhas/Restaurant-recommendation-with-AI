from random import choice 
from numpy import array, dot, random 
import numpy as np

training_data = [ 
				  (array([1,0,0]),array([0,1,0]),array([0,0,1]),"dislike"),(array([0,0,1]),array([1,0,0]),array([0,1,0]),"like"),
                  (array([0,1,0]),array([0,0,1]),array([0,0,1]),"dislike"),(array([0,1,1]),array([0,1,0]),array([1,0,0]),"like"),
				  (array([1,1,0]),array([0,1,0]),array([1,0,0]),"dislike"),(array([1,1,1]),array([0,1,0]),array([1,0,0]),"like"),
				  (array([1,1,0]),array([0,1,0]),array([1,0,0]),"dislike"),(array([0,1,1]),array([0,1,0]),array([1,0,0]),"like"),
				  (array([1,1,0]),array([0,1,0]),array([1,0,0]),"dislike"),(array([1,0,1]),array([0,1,0]),array([1,0,0]),"like"),
				  (array([1,1,0]),array([0,1,0]),array([1,0,0]),"dislike"),(array([0,0,1]),array([0,1,0]),array([1,0,0]),"like"),
				  (array([1,1,0]),array([0,1,0]),array([1,0,0]),"dislike"),(array([1,1,1]),array([0,1,0]),array([1,0,0]),"like"),
				  (array([1,1,0]),array([0,1,0]),array([1,0,0]),"dislike"),(array([0,1,1]),array([0,1,0]),array([1,0,0]),"like"),
				  (array([1,1,0]),array([0,1,0]),array([1,0,0]),"dislike"),(array([1,0,1]),array([0,1,0]),array([1,0,0]),"like"),
				  (array([1,1,0]),array([0,1,0]),array([1,0,0]),"dislike"),(array([0,0,1]),array([0,1,0]),array([1,0,0]),"like"),
				  (array([1,1,0]),array([0,1,0]),array([1,0,0]),"dislike")
				  ]

training_data1 = [ 
				  (array([1,0,0]),array([0,0,1]),array([0,0,1]),"like"),(array([0,0,1]),array([1,0,0]),array([0,1,0]),"dislike"),
                  (array([1,0,0]),array([0,0,1]),array([0,1,1]),"like"),(array([0,1,1]),array([0,1,0]),array([1,0,0]),"dislike"),
				  (array([1,0,0]),array([0,1,0]),array([1,0,0]),"like"),(array([1,1,1]),array([0,1,0]),array([1,0,0]),"dislike"),
				  (array([1,0,1]),array([0,0,1]),array([1,0,0]),"like"),(array([0,1,1]),array([0,1,0]),array([1,0,0]),"dislike"),
				  (array([1,1,0]),array([0,1,0]),array([0,0,1]),"like"),(array([1,0,1]),array([0,1,0]),array([1,0,0]),"dislike"),
				  (array([1,1,0]),array([0,1,0]),array([0,0,1]),"like"),(array([0,0,1]),array([0,1,0]),array([1,0,0]),"dislike"),
				  (array([1,0,0]),array([0,1,0]),array([0,1,1]),"like"),(array([1,1,1]),array([0,1,0]),array([1,0,0]),"dislike"),
				  (array([1,0,0]),array([0,0,1]),array([1,0,0]),"like"),(array([0,1,1]),array([0,1,0]),array([1,0,0]),"dislike"),
				  (array([1,1,1]),array([0,0,1]),array([1,0,0]),"like"),(array([1,0,1]),array([0,1,0]),array([1,0,0]),"dislike"),
				  (array([1,0,1]),array([0,0,1]),array([0,0,1]),"like"),(array([0,0,1]),array([0,1,0]),array([1,0,0]),"dislike"),
				  (array([1,0,0]),array([0,1,0]),array([0,0,1]),"like")
				  ]
				  
test_data = [ 
				  (array([1,0,0]),array([0,1,0]),array([0,1,0]),"dislike"),
                  (array([0,1,0]),array([0,0,1]),array([0,0,1]),"dislike"),
				  (array([0,0,1]),array([1,0,0]),array([0,1,0]),"like"),
				  (array([1,1,0]),array([0,1,0]),array([0,1,0]),"dislike"),
				  (array([0,1,1]),array([1,1,0]),array([0,1,0]),"like")
				  ]

test_data1 = [ 
				  (array([1,0,0]),array([0,0,1]),array([0,0,1]),"like"),
                  (array([0,0,1]),array([0,1,0]),array([1,0,0]),"dislike"),
				  (array([0,0,1]),array([0,1,0]),array([1,0,0]),"dislike"),
				  (array([1,1,0]),array([0,1,0]),array([0,1,0]),"dislike"),
				  (array([1,0,0]),array([0,1,0]),array([0,0,1]),"like"),
				  (array([1,0,0]),array([0,0,1]),array([0,0,1]),"like"),
                  (array([0,0,1]),array([0,1,0]),array([1,0,0]),"dislike"),
				  (array([0,0,1]),array([0,1,0]),array([1,0,0]),"dislike"),
				  (array([1,0,0]),array([0,1,0]),array([0,0,1]),"like")
				  ]
				  
w = [ array([0,0,0]),array([0,0,0]),array([0,0,0]) ]

#unit_step = lambda x: 0 if x < 0 else 1 
#features cuisine[0Chinese 0American 1Indian] cost[0lessThan20 1between20And30 0GreaterThan30] happyhours[0Nope 1Sometimes 1AllDay]
#         Take-out,DineIn, drinks[] dessert,fastFood,Healthy[]
def dot_product(a, b):
    return sum([a[i]*b[i] for i in range(len(a))])

def addWeightFeature(a, b):
    return ([a[i]+b[i] for i in range(len(a))])
	
def subWeightFeature(a, b):
    return ([b[i]-a[i] for i in range(len(a))])	

def test_accuracy(w):
	print"******************************RECOMMENDATION STARTS**********************************************"
	count=0
	for i in range(len(test_data1)):
		features=test_data1[i]
		iter=range(len(w))
		score=0
		
		for fLen in iter:
			#print "tdata",training_data[i][fLen]
			#print  "wdata",w[fLen]
			score+=dot_product(test_data1[i][fLen], w[fLen])
		#print"score",score
		if(score >= 0):
				print "Recommend Restaurant ",i+1
				count+=1
		


for i in range(len(training_data1)):
	features=training_data1[i]
	#weights=w[0]
	#print len(w),w[0],w[1],w[2]
	score=0
	iter = range(len(w))
	neww=[]
	
	for fLen in iter:
		#print "tdata",training_data[i][fLen]
		#print  "wdata",w[fLen]
		score+=dot_product(training_data1[i][fLen], w[fLen])
	# if(score >= 0):
				# label="like"
	# else:
				# label="dislike "
	
	if(training_data1[i][fLen+1] == "like"):
	#if(training_data[i][fLen+1] == label):
		print"dummy"
		for fLen in range(len(w)):
			 neww.append(array(addWeightFeature(training_data1[i][fLen],w[fLen])))
		w=neww
		# print "wAfterLike",w
	else:
		#print"dislike"
		for fLen in range(len(w)):
			neww.append(array(subWeightFeature(training_data1[i][fLen],w[fLen])))
		w=neww
		#print "wAfterDislike",w
print"******************************Converged hyperplane**********************************************"
print "Final hyperplane",w
test_accuracy(w)


			