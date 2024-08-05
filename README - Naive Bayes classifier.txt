Name: Priyanka Pradhan
Naïve Bayesian Classification 
Version – VS Python 3.9

objective :By calculating the probability determine whether a radar target is a bird or an aircraft .
Given -
data.txt with containing 10  tracks  representing  the  velocity  of the object at a given time 
pdf.txt - probabilty density function containing the probability of being an airplane or bird based on speed

Assumptions - 
* From the pdf value in pdf.txt , out of 2 rows the first row belongs to the pdf of bird ( considering lots of 0s at the end) and second row belongs to airplane 
as Most aircraft travel faster than most birds, but some overlap exists, especially with smaller, lower-performance aircraft .
* As shown in the graph speed goes till 0 to 200 , and the given pdf.txt has 400 values in it . the velocity value for an object at a single point of time
was doubled to get the probability , so that for each 2 pdfs it matches with 1 speed window.
* took prior= 0.5 as  initial  probabilities  for  the  classes,so that classification is equally distributed.Thus used for t=0 for both bird and plane
* replaced the values of NaNs with the mean value of the rows 
* the classifier is conservative when transitioning between classes of objects , the probabilty of being a bird given it was a bird earlier = 0.9,
P(airplane|airplane)= 0.9, P(bird|airplane)= 0.1 ,P(airplane|bird)= 0.1 . These values can change slightly but to be on the safer side we took the highest possibility.


About the Output:
* show all 300 probabilities for bird and airplane for all the 10 tracks.( will loop for 10 times)
* calculate the total sum of probabilities for bird and plane
* determine and show which track is likely to be a bird or a aircraft.

Additonal features to improve the classification -
Naive Bayesian classifier inputs discrete variables and outputs a probability score for each candidate class.
The predicted class label is the class label with the highest probability score. The probabilities obtained in these observations provided were quite distinctive.
Few things can be done but I believe won't make much difference ,
*To avoid working with very small numbers, we can work within the log probability space by taking the logarithm of probability values.
* eliminate the zero observations, Applying a smoothing technique assigns a very small probability estimate to such zero frequency occurrences to regularize.









