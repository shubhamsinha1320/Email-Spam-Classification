# Spam-Classification
Email spam classification and prediction whether a newly arrived mail is a spam or not.


It is a email spam classification project where a number of mails were used for training and testing.
The dataset used for this purpose was 'fraud_email_.csv' which is also present in the repository.

There are some popular python libraries which were used during the development of program such as, pandas, tensorflow, nltk, sklearn and csv.
The model I used for spam detection is Natural Language Processing's bag of words model and the accuracy of the model is found 99.1 % (approximately).

There are two developer-made functions in the program- message_preprocessing() and add_message_to_dataset().

message_preprocessing() - In this function, the message entered by the user is moulded the same way as the training and testing data of the program.
add_message_to_dataset() - The only purpose of this function is to add the latest message to the dataset. This will increase the range of the program and help increase the accuracy of the program.
