import os
import extract_msg
import csv
import re

mkdir = 'mkdir ./train-uploads ./train-uploads/extracted-data'
unzip = 'unzip ./train-uploads/*.zip -d ./train-uploads/extracted-data'
rmzip = 'rm ./train-uploads/*.zip'
rmmsg = 'rm -r ./train-uploads/'

def cleanForTrain():
    i = 1
    os.system(mkdir)
    os.system(unzip)
    os.system(rmzip)
    with open(os.path.join(os.getcwd(), 'test-uploads','model-input','training.csv'), 'wt') as file:
        fieldnames = ['Filename', 'uid', 'Subject', 'Date', 'Sender', 'Body', 'Body_Unformatted', 'Label']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for folder in os.listdir(os.path.join(os.getcwd(), 'train-uploads', 'extracted-data')):
            for f in os.listdir(os.path.join(os.getcwd(),'train-uploads','extracted-data', folder)):
                if not f.endswith('.msg'):
                    continue
                msg = extract_msg.Message(os.path.join(os.getcwd(),'train-uploads','extracted-data', folder, f))
                msg_sender = msg.body
                msg_date = msg.date
                msg_subj = msg.subject
                msg_message = msg.body
                msg_sender = re.findall('From *: (.+)\n', msg_sender)
                msg_message = re.sub('From *: (.*)\n','', msg_message)
                msg_message = re.sub('To *: (.*)\n','', msg_message)
                msg_message = re.sub('Cc *: (.*)\n','', msg_message)
                msg_message = re.sub('Sent *: (.*)\n','', msg_message)
                msg_message = re.sub('Subject *:','', msg_message)
                msg_uformatted = msg_message
                msg_message = re.sub('[^a-zA-z,\.]'," ",msg_message)

                msg_message = ' '.join(re.split("\n",msg_message))
                msg_message = ' '.join(re.split(" +",msg_message))
                msg_message = ''.join(re.split("\r",msg_message))

                writer.writerow({'Filename': f, 'uid': str(i), 'Subject': msg_subj, 'Date': msg_date, 'Sender': msg_sender[0], 'Body': msg_message.encode('utf-8'), 'Body_Unformatted': msg_uformatted.encode('utf-8'), 'Label': folder})
                i += 1
        os.system(rmmsg)
