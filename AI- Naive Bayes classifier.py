
import pandas as pd
import math as m
import numpy as np


### load the records from text files having probabilty density func (pdf) and velocities at a time interval of 0.5(data)
pdf = pd.read_csv('pdf.txt', sep=",", header=None)
data = pd.read_csv('data.txt', sep=",", header=None)
##print(pdf)
##print(data)

## start  equally  distributed values
prior = 0.5

##replace NAN values  with mean of objects
for i in range(0,10):
    #data.loc[i] = data.loc[i].fillna(0)
    data.loc[i] = data.loc[i].fillna(value=data.loc[i].mean())

 ## defining a class for the calculating the naive bayes and getting the probabilities of bird and aircraft
class NaiveBayes:
   
    def __init__(self, pdf, data, val):
        ## empty list to store values from pdf.txt and data.txt
        pdf= [] 
        data = []
        ## values of all 10 rows 
        self.val = val

    def get_velocity_prob(self, time):
       ##getting velocities for each time frame
        velocity = data.iloc[self.val][time]
        each_velocity = int(velocity) * 2

      ##  assuming the row 1 of pdf is of  bird
        pdf_bird = pdf[each_velocity][0]
        ## assuming row 2 of pdf  is of aircraft
        pdf_airplane = pdf[each_velocity][1] 

        return pdf_bird, pdf_airplane


    #probability of only getting bird at a given time
    def probability_bird(self, time):
        bird = self.get_velocity_prob(time)[0]
        return bird 

      #probability of only getting airplane at a given time
    def probability_airplane(self, time):
        airplane = self.get_velocity_prob(time)[1]
        return airplane 

    def RecursiveBayesianEstimation(self):
      
        #create empty lists which will store all 300 probabilties for bird and airplane
        all_bird_probab = []
        all_airplane_probab = []

      
        #find out belief  at t = 0
       ## 𝐵0 𝑗 =𝑃 𝑜0 𝑗)𝜋0(𝑗) ∀𝑗∈𝑆
       ##𝑏0 = Normalize 𝐵0
        belief_bird0 = self.probability_bird(0) * prior 
        belief_airplane0 = self.probability_airplane(0) * prior 
        #normalize the belief of bird and plane to get the probabilty of bird and airplane at time = 0
        
        prob_bird_0 = belief_bird0/(belief_bird0 + belief_airplane0)
        prob_airplane_0 = belief_airplane0/(belief_bird0 + belief_airplane0) 

        #append to list
        all_bird_probab.append(prob_bird_0)
        all_airplane_probab.append(prob_airplane_0)

        ##at remaining t's
       ## for t = 1 to 𝑻
        ##𝐵𝑡 𝑗 =𝐿 𝑜𝑡 𝑗)σ𝑠𝑃 𝑗 𝑠)𝑏𝑡−1(𝑠) ∀𝑗,𝑠∈𝑆
        ##𝑏𝑡 = Normalize 𝐵𝑡
        for t in range(1,300):
            #find out belief when state is bird at  t
            bird_t = self.probability_bird(t) 
            ##prob its a bird given it was a bird and prob that its a airplane given it was a bird earlier(past)
            prob_bird_given_bird = 0.9 * all_bird_probab[t-1]
            prob_airplane_given_bird = 0.1 * all_airplane_probab[t-1]

            Likelyhood_bird_given_velocity = prob_bird_given_bird + prob_airplane_given_bird
           ##probability of being a bird
            being_bird = bird_t * Likelyhood_bird_given_velocity

            #find out belief when state is plane at t
            airplane_t = self.probability_airplane(t) 
             ##prob its a airplane given it was a airplane and prob that its a bird given it was a airplane earlier(past)
            prob_airplane_given_airplane = 0.9 * all_airplane_probab[t-1]
            prob_bird_given_airplane = 0.1 * all_bird_probab[t-1]

            likelyhood_airplane_given_velocity = prob_airplane_given_airplane + prob_bird_given_airplane  
            ##probability of being a airplane
            being_airplane = airplane_t * likelyhood_airplane_given_velocity

            #normalize
            Total_sum_bird_plane = being_bird + being_airplane
            #get probs for each 
            prob_bird_t = being_bird / Total_sum_bird_plane
            prob_airplane_t = being_airplane / Total_sum_bird_plane

            #attend probs for each t
            all_bird_probab.append(prob_bird_t)
            all_airplane_probab.append(prob_airplane_t)

        return all_bird_probab, all_airplane_probab


class RBE(NaiveBayes):
    def prob_bird_plane(self):
       

        prob_list_b = self.RecursiveBayesianEstimation()[0] #prob list for bird
        prob_list_p = self.RecursiveBayesianEstimation()[1] #prob list for plane

        #data frame creation
        bird_plane_frame = pd.DataFrame((zip(prob_list_b, prob_list_p)),
                         columns=['P(Bird)', 'P(Plane)'])
        bird_sum = sum(prob_list_b)
        airplane_sum = sum(prob_list_p)
        print('Total sum of P(Birds):', bird_sum)
        print( 'Total sum of P(airplane):' , airplane_sum ,'\n')
        ##find which track belongs to which object bird / plane
        if bird_sum > airplane_sum :
            print("Conclusion: This track belongs to BIRDS \n")
        else:
            print("Conclusion: This track belongs to AIRPLANE \n")
        
        return bird_plane_frame
for tracks in range(0,10): 
            navbay = RBE(pdf, data, tracks)
            print("probabilities of track number:" ,tracks + 1)
            print( )
            print(navbay.prob_bird_plane())
            
                     

