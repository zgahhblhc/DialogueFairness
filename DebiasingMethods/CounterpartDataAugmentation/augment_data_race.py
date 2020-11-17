import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import random
import re
ori_train_file = 'data/Twitter/train.txt'
out_dir = 'augmented_data/Twitter/'

white_to_black, black_to_white = {}, {}
with open('words/race_words.txt', 'r') as f:
    word_lines = f.readlines()
    for line in word_lines:
        white_word, black_word = line.strip().split(' - ')
        white_word = white_word.lower()
        black_word = black_word.lower()
        if white_word in white_to_black:
            white_to_black[white_word].append(black_word)
        else:
            white_to_black[white_word] = [black_word]
        if black_word in black_to_white:
            black_to_white[black_word].append(white_word)
        else:
            black_to_white[black_word] = [white_word]

print(white_to_black, black_to_white)

def create_parallel(text):
    tmp_new_text = text

    black_english = False

    for w in black_to_white:
        new_word = random.choice(black_to_white[w])
        s = re.sub(r'(?<=\b)' + w + r'(?=\b)', new_word, tmp_new_text)
        if s != tmp_new_text:
            tmp_new_text = s
            black_english = True

    if not black_english:
        for w in white_to_black:
            new_word = random.choice(white_to_black[w])
            s = re.sub(r'(?<=\b)' + w + r'(?=\b)', new_word, tmp_new_text)
            if s != tmp_new_text:
                tmp_new_text = s

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