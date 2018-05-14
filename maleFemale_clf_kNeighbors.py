from sklearn.neighbors import KNeighborsClassifier


clf = KNeighborsClassifier(n_neighbors=3)

#[height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

clf.fit(X, Y)

# get a input from user
# comment this if user input not needed
user_input = raw_input("enter height, weight, shoe_size as comma separated as => 190, 70, 43 \n >").split(',') #will take in a string of numbers separated by a space
user_input = [int(num) for num in user_input] 

# uncomment this if user input not needed
# user_input = [190, 70, 43]

prediction = clf.predict([user_input])

print(prediction)