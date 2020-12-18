import os
import extract_msg
import csv
import re

mkdir = 'mkdir ./test-uploads/extracted-data'
unzip = 'unzip -j ./test-uploads/*.zip -d ./test-uploads/extracted-data'
rmzip = 'rm ./test-uploads/*.zip'
rmmsg = 'rm -rf ./test-uploads/extracted-data/*'

def cleanForTest():
    i = 1
    os.system(mkdir)
    os.system(unzip)
    os.system(rmzip)
    with open(os.path.join(os.getcwd(), 'test-uploads','model-input','testing.csv'), 'wt') as file:
        fieldnames = ['Filename','uid', 'Subject', 'Date', 'Sender', 'Body', 'Body_Unformatted']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for f in sorted(os.listdir(os.path.join(os.getcwd(),'test-uploads','extracted-data'))):
            if not f.endswith('.msg'):
                continue
            msg = extract_msg.Message(os.path.join(os.getcwd(),'test-uploads','extracted-data', f))
            msg_sender = msg.body
            msg_date = msg.date
            msg_subj = msg.subject
            msg_message = msg.body
            # fh = open(os.path.join(folder, f), 'a+', encoding='utf8', errors='ignore')
            # for line in fh.read():
            #     line = re.sub('[^a-zA-z]'," ",line)
            #     if 'From:' not in line and 'To:' not in line and 'Cc:' not in line and 'Cc :' not in line and 'To :' not in line and 'From :' not in line and 'Sent:' not in line and 'Sent :' not in line:
            #         fh.write(line)
            #     else:
            #         continue
            # fh.close()

            msg_sender = re.findall('From *: (.+)\n', msg_sender)
            msg_message = re.sub('From *: (.*)\n','', msg_message)
            msg_message = re.sub('To *: (.*)\n','', msg_message)
            msg_message = re.sub('Cc *: (.*)\n','', msg_message)
            msg_message = re.sub('Sent *: (.*)\n','', msg_message)
            msg_message = re.sub('Subject *:','', msg_message)
            msg_uformatted = msg.body
            msg_message = re.sub('[^a-zA-z,\.]'," ",msg_message)
            msg_message = ' '.join(re.split("\n",msg_message))
            msg_message = ' '.join(re.split(" +",msg_message))
            msg_message = ''.join(re.split("\r",msg_message))

            writer.writerow({'Filename': f, 'uid': str(i), 'Subject': msg_subj, 'Date': msg_date, 'Sender': msg_sender[0], 'Body': msg_message.encode('utf-8'), 'Body_Unformatted': msg_uformatted.encode('utf-8')})
            i += 1
        os.system(rmmsg)
