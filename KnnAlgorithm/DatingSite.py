import csv
# I am going to use it as my homework assignment 
# Task 1: Read from a file 
# Task 2: Parse it in python
# Task 3: plotting
# Task 4: code the classifier
# Task 5: one line function call for helen 

filePath = "D:\GithubRepositories\MachineLearningPythonCodes\KnnAlgorithm\datingTestSet.txt"

# first problem - can't use .read() as it stores everyting(\t) as a string
# so can't use fileVariable = open(filePath,mode='r')
# Solution :- csv 

with open(filePath) as file :
    reader = csv.reader(file,delimiter="\t")
    d = list(reader)
    # stores as a list and len() is used to find the length 

# Inorder to plot it, I want to first store each column in a different variable 
# and convert them to the required datatype 


