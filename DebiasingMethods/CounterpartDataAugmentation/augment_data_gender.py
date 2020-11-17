import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import random
ori_train_file = 'data/Twitter/train.txt'
out_dir = 'augmented_data/Twitter/'

male_to_female, female_to_male = {}, {}
with open('words/gender_words.txt', 'r') as f:
    word_lines = f.readlines()
    for line in word_lines:
        male_word, female_word = line.strip().split(' - ')
        male_to_female[male_word] = female_word
        female_to_male[female_word] = male_word

def create_parallel(text):
    words = text.split()
    new_text_words = []

    for word in words:
        if word in male_to_female:
            new_text_words.append(male_to_female[word])
        elif word in female_to_male:
            new_text_words.append(female_to_male[word])
        else:
            new_text_words.append(word)

    tmp_new_text = ' '.join(new_text_words)

    new_text = tmp_new_text

    return new_text

aug_num = 0
dialogue_num = 0
new_dialogues = []

with open(ori_train_file, 'r') as f:
    lines = f.readlines()
    count = 0
    for line in lines:
        new_dialogues.append(line.strip())
        count += 1
        if count % 10000 == 0:
            logger.info(count)

        texts = line.strip().split('\t')
        assert len(texts) == 3
        text, labels, episode_done = texts
        text_content = text[5:]
        label_content = labels[7:]

        par_text = create_parallel(text_content)

        if par_text != text_content:
            aug_num += 1
            par_labels = create_parallel(label_content)

            par_dialog = 'text:' + par_text + '\t' + 'labels:' + par_labels + '\t' + episode_done
            new_dialogues.append(par_dialog)

        if len(new_dialogues) >= 20000:
            random.shuffle(new_dialogues)
            with open(out_dir + 'train.txt', 'a+') as f:
                f.write('\n')
                f.write('\n'.join(new_dialogues))
                logger.info("Add to train.txt")
            dialogue_num += len(new_dialogues)
            new_dialogues = []

if len(new_dialogues) > 0:
    random.shuffle(new_dialogues)
    with open(out_dir + 'train.txt', 'a+') as f:
        f.write('\n')
        f.write('\n'.join(new_dialogues))
        logger.info("Add to train.txt")
    dialogue_num += len(new_dialogues)
    new_dialogues = []

print("aug_num: ", aug_num)
print("new dialogue length: ", dialogue_num)