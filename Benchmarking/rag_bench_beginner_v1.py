from datasets import load_dataset
import evaluate
import ollama
import time
from csv import DictWriter
from sentence_transformers import SentenceTransformer,util
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('ggplot')
start = time.time()

################  Hyper Parameters ###################
model_id = 'gemma:2b'
dataset_name = 'databricks/databricks-dolly-15k'
n = 100         # number of evaluation examples
embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

#################################################
rouge = evaluate.load('rouge')

model = SentenceTransformer(embed_model_id)

dataset = load_dataset(dataset_name,split='train')

def generate_response(query,context):
    response = ollama.chat(model=model_id, messages=[
    {
        'role': 'user',
        'content': f'Answer query given the context. Keep the answer as short as possible.\n{query}\n{context}',
    },
    ])
    result = str(response['message']['content'])
    return result

def evaluate(dataset):
    score = 0
    gen_text,y = [],[]
    for i in tqdm(range(0,n),desc="Progress"):
        pred = generate_response(dataset[i]['instruction'],dataset[i]['context'])
        gen_text.append(pred)
        pred_encoding = model.encode(pred)
        true_encoding = model.encode(dataset[i]['response'])
        #print('\n\nExpected Response: {}\n\nGenerated Response: {}'.format(dataset[i]['response'],pred))
        cos_sim = util.cos_sim(true_encoding,pred_encoding)
        y.append(cos_sim.item())
        #print('\nCosine simialrity:{}'.format(cos_sim))
        if cos_sim > 0.5:
            score += 1
    score /= n

    #print('Score at itreation-{} : {}'.format(i,score))
    return score,gen_text,y

def update_csv(output):
    field_names = ['Model','Dataset Id','Accuracy','Cos_avg','Rouge Scores','Num_samples','Time']
    with open('event.csv', 'a') as f_object:
        dictwriter_object = DictWriter(f_object, fieldnames=field_names)
        dictwriter_object.writerow(output)
        f_object.close()
        
def main(): 
    score,gen_text,y = evaluate(dataset)    
    #print('Accuracy of Model (based on averaged cosine similarities) :',score)
    rouge_score = rouge.compute(predictions=gen_text,
                                references=dataset['response'][:n])
    #print('Rouge score:',rouge_score)       
    cos_avg = sum(y)/n
    end = time.time()
    print(f'Time of execution : {(end-start):.4f} seconds')
    out_data = {"Model":model_id,
                "Dataset Id":dataset_name,
                "Accuracy":score,
                "Cos_avg": cos_avg,
                'Rouge Scores':rouge_score,
                'Num_samples':n,
                'Time':end-start
                }
    update_csv(out_data)
    plt.title('Cosine_Similarity Vs Examples')
    plt.xlabel('Examples')
    plt.ylabel('Cosine Similarity')
    plt.plot(range(0,n),y,color='mediumslateblue')
    plt.ylim([0, 1])
    plt.show()
    
if __name__ =='__main__':
    main()
