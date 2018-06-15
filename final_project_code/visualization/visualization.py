#visualsation of matrix
import numpy as np
import matplotlib.pyplot as plt 

svm_confusion = np.asarray([[0.53404324, 0.11648919, 0.34946757],
       [0.14515446, 0.61362675, 0.24121879],
       [0.08869035, 0.0343212 , 0.87698845]])
lr_confusion = np.asarray([[0.53662472, 0.10487254, 0.35850274],
       [0.14261532, 0.60854846, 0.24883623],
       [0.08498584, 0.029854  , 0.88516017]])
cnn_confusion = np.asarray([[0.94045471, 0.01515698, 0.04438831],
       [0.72262774, 0.02773723, 0.24963504],
       [0.74893162, 0.06410256, 0.18696581]])

def visualize(matrix, dataset_title, label_class,x_label_text = 'Predicted Lables',y_label_text = 'Actual Labels'):
  fig,ax = plt.subplots()
	# round the accuracy in matrix after 2 digits after dot
  show_matrix = np.around((matrix), decimals=2)
  im = ax.imshow(show_matrix)
  #set the ticks
  ax.set_xticks(np.arange(len(label_class)))
  ax.set_yticks(np.arange(len(label_class)))
  ax.set_xticklabels(label_class)
  ax.set_yticklabels(label_class)
	#rotate the tick labels and set alignments
  plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
	#set title
  ax.set_title(dataset_title)
	#set x label
  ax.set_xlabel(x_label_text)
  ax.set_ylabel(y_label_text)
	#create text annotation
  for x in range(len(label_class)):
    for y in range(len(label_class)):
      text = ax.text(y, x, show_matrix[x, y], ha='center', va='center', color='w')
  #create color bar
  cbar = ax.figure.colorbar(im, ax=ax)
  #cbar = ax.set_ylabel()
  fig.tight_layout()
  plt.show()

#run it 
title_svm = 'Normalized Confusion Matrix of Unigram Based on SVM'
title_lr = 'Normalized Confusion Matrix of Unigram Based on Logistic Regression'
title_cnn = 'Normalized Confusion Matrix of Unigram Based on CNN'
labels = np.asarray(['neutral', 'positive', 'Negative'])
#emotic_labels = np.asarray(['angry', 'disgust', 'fear','happy', 'sad', 'surprise'])
#visualize(svm_confusion, title_svm, labels)
visualize(lr_confusion, title_lr, labels)
#visualize(cnn_confusion, title_cnn, labels)
#visualize(emotic, title_emotic, emotic_labels)
#visualize(ck, title_ck, labels)
#print(np.around(private, decimals=2))