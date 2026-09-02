"""Vendored Arabic diacritization evaluator from Misraj (Sadeed).

Source: https://github.com/misraj-ai/Sadeed Eval_code.ipynb (cell 3),
MIT-style released alongside the SadeedDiac-25 benchmark
(https://huggingface.co/datasets/Misraj/SadeedDiac-25).
Unmodified except for this header.

Metrics: Total DER (= DER with case endings), Morphological DER
(= DER without case endings, word-final letter excluded), plus WERs.
"""

from pyarabic import araby
from prettytable import PrettyTable
import re
from tqdm import tqdm
import warnings
import pandas as pd

class ArabicDiacritizationEvaluator:
    """
    A utility class for evaluating Arabic diacritization,
    """
    STANDARD_ARABIC_LETTER_PATTERN = r"ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي"
    STANDARD_HARAKA_PATTERN = r'ًٌٍَُِّْٓ' #harakat, madda and shadda
    TATWEEL_PATTERN = r"ـ"
    QURANIC_AND_ISLAMIC_ANNOTATION_SIGNS = r'ؘؙؚٰؐؑؒؓؕؗ٘ۖۗۘۙۚۛۜٱ۝۞ۣ۟۟۠ۡۢۤۥۦۧۨ۩۪ۭٕ۫۬۬ٗٔࣔࣕࣖࣗࣘࣙࣚࣛࣜࣝࣞࣟ࣠࣡࣢ࣰࣱࣲࣳ﴾﴿ﷰﷱﷲﷳﷳﷴﷵﷶﷷﷸﷺﷻ﷼﷽'
    NON_ARABIC_SYMOBOLS_IN_ARABIC_BLOCK = r"؀؁؂؃؄؅؎؏ؘؙؚؔؖ؋"
    NON_ARABIC_LETTERS_IN_ARABIC_UNICODE_BLOCK = r'ؠػؼؽؾؿٮٯٲٳٴٵٶٷٸٹٺٻټٽپٿڀځڂڃڄڅچڇڈډڊڋڌڍڎڏڐڑڒړڔڕږڗژڙښڛڜڝڞڟڠڡڢڣڤڥڦڧڨکڪګڬڭڮگڰڱڲڳڴڵڶڷڸڹںڻڼڽھڿۀہۂۃۄۅۆۇۈۉۊۋیۍێۏېۑےۓەۮۯۺۻۼ۽۾ۿ'
    MODOUD_LETTERS = [araby.ALEF, araby.ALEF_MADDA, araby.ALEF_WASLA,araby.ALEF_MAKSURA,araby.SMALL_ALEF, araby.WAW, araby.SMALL_WAW, araby.YEH, araby.SMALL_YEH]

    NO_HARAKA = '*'
    ARABIC_LETTERS_PATTERN = r"["+STANDARD_ARABIC_LETTER_PATTERN+"]+"

    FULLY_EXTENED_ARABIC_WORD = (
        STANDARD_ARABIC_LETTER_PATTERN +
        STANDARD_HARAKA_PATTERN +
        TATWEEL_PATTERN +
        QURANIC_AND_ISLAMIC_ANNOTATION_SIGNS +
        NON_ARABIC_SYMOBOLS_IN_ARABIC_BLOCK +
        NON_ARABIC_LETTERS_IN_ARABIC_UNICODE_BLOCK)


    @classmethod
    def split_arabic_text(cls, text: str) -> list:
        """
        Splits Arabic text into words and symbols based on extended Arabic characters.
        """
        det_chars = cls.FULLY_EXTENED_ARABIC_WORD
        pattern = f"([{re.escape(det_chars)}]+)"
        result = re.split(pattern, text)
        result = [word for word in result if word] # Remove empty strings
        return result

    @classmethod
    def extract_harakat(cls, word: str, shadda_is_letter: bool = False):
        """
        Extracts harakat and corresponding letters from a diacritized word.
        returns harakat , letters, only one haraka is allowed
        """
        harakat = []
        letters = []
        i = -1
        while(word != ""):
            char = word[0]
            word = word[1:]
            if araby.is_haraka(char) or araby.is_shadda(char):
                if len(letters)>0: #pass harakat before word
                    if char == araby.SHADDA:
                        if shadda_is_letter:
                            if letters[i].endswith(araby.SHADDA): # there is already a shadda
                                pass
                            else:
                                letters[i] = letters[i] + araby.SHADDA
                        else:
                            if (harakat[i] == cls.NO_HARAKA):
                                harakat[i] = char
                    else: # not shadda
                        if (harakat[i] == cls.NO_HARAKA):
                            harakat[i] = char
                        elif harakat[i] == araby.SHADDA:
                            harakat[i] = harakat[i] + char #only allow one haraka after shadda
                        else:
                            pass #ignore
                else: pass#pass harakat before word
            else: # it is letter
                letters.append(char)
                harakat.append(cls.NO_HARAKA)
                i += 1

        return harakat, letters


    @classmethod
    def has_arabic_letter(cls, str):
        result =  re.findall(cls.ARABIC_LETTERS_PATTERN, str)
        return (not result == None and not result  == [])

    @classmethod
    def has_al_alta3reef(cls, word: str, must_have_voweles: bool = False):
        '''
        checks to see if the word has al alta3reef
        this fucntion may give errors if the word starts with something that looks like al alta3reef الزم
        @returns true or false + the letter order of the begining of al alt3reef (if there are prefixes this will not
        be 0, else it will be 0) or -1 if no al alt3reef was found

        '''

        harakat, letters = cls.extract_harakat(word)
        if len(letters) < 4:
            return False, -1
        #
        no_harak_or_sukun = [cls.NO_HARAKA , araby.SUKUN]
        no_harak_or_fatha = [cls.NO_HARAKA , araby.FATHA]
        no_harak_or_kasra = [cls.NO_HARAKA , araby.KASRA]
        # check the prefix letters before the word that can take al alt3reef after
        # وَ   فَ    لِ   أَ    بِ    كَ    لً    سَ
        index = 0
        if must_have_voweles:
            while (True):
                if index >= len(letters):
                    return False , -1
                if letters[index] == araby.WAW and harakat[index] == araby.FATHA: # وَ واو العطف
                    index += 1
                elif letters[index] == araby.FEH and harakat[index] == araby.FATHA: #فَ فاء العطف
                    index += 1
                elif letters[index] == araby.BEH and harakat[index] == araby.KASRA: # بِ باء الجر
                    index += 1
                elif letters[index] == araby.KAF and harakat[index] == araby.FATHA: # كَ كاف التشبيه
                    index += 1
                elif letters[index] == araby.ALEF_HAMZA_ABOVE and harakat[index] == araby.FATHA: # أ همزة إستفهام
                    index += 1
                # لا تأتي مع ال التعريف
                # elif letters[index] == araby.SEEN and harakat[index] in no_harak_or_fatha: # سَ سين المستقبل
                #     index += 1
                else:
                    break
        else: # don't check vowles
            while (True):
                if index >= len(letters):
                    return False , -1
                if letters[index] == araby.WAW and harakat[index] in no_harak_or_fatha: # وَ واو العطف
                    index += 1
                elif letters[index] == araby.FEH and harakat[index] in no_harak_or_fatha: #فَ فاء العطف
                    index += 1
                elif letters[index] == araby.BEH and harakat[index] in no_harak_or_kasra: # بِ باء الجر
                    index += 1
                elif letters[index] == araby.KAF and harakat[index] in no_harak_or_fatha: # كَ كاف التشبيه
                    index += 1
                elif letters[index] == araby.ALEF_HAMZA_ABOVE and harakat[index] in no_harak_or_fatha: # أ همزة إستفهام
                    index += 1
                # لا تأتي مع ال التعريف
                # elif letters[index] == araby.SEEN and harakat[index] in no_harak_or_fatha: # سَ سين المستقبل
                #     index += 1
                else:
                    break


        # there must be letters left because it didn't return
        letters = letters[index:]
        harakat = harakat[index:]

        if len(letters) < 4: #السَمّ أقل شي حرفين بعد أل التعريف
            return False, -1

        next_letter_haraka = harakat[2]
        if letters[0] == araby.ALEF and letters[1] == araby.LAM:
            if harakat[0] == cls.NO_HARAKA:
                if (letters[2] == araby.ALEF and harakat[1] == araby.KASRA): #حالة الِانْدِلَاعِ عليها كسرة
                    return True, index
                if must_have_voweles:
                    # الشمسي ما في عاللام حركة والقمري عاللام سكون
                    if (harakat[1] == cls.NO_HARAKA and araby.SHADDA in next_letter_haraka) or\
                        (harakat[1] == araby.SUKUN and not araby.SHADDA in next_letter_haraka):
                        return True, index
                else:
                    if harakat[1] in no_harak_or_sukun:
                        return True, index
        return False ,-1


    #لِلِاسْتِفَادَةِ لَلسَّماء
    @classmethod
    def has_ll_alta3reef(cls, word: str, must_have_voweles: bool = False):
        harakat, letters = cls.extract_harakat(word)
        if len(letters) < 4:
            return False, -1
        no_harak_or_sukun = [cls.NO_HARAKA , araby.SUKUN]
        no_harak_or_fatha = [cls.NO_HARAKA , araby.FATHA]
        no_harak_or_kasra_or_fatha = [cls.NO_HARAKA , araby.KASRA, araby.FATHA]
        kasra_or_fatha = [araby.KASRA, araby.FATHA]
        index = 0
        if must_have_voweles:
            while (True):
                if index >= len(letters):
                    return False , -1
                if letters[index] == araby.WAW and harakat[index] == araby.FATHA: # وَ واو العطف
                    index += 1
                elif letters[index] == araby.FEH and harakat[index] == araby.FATHA: #فَ فاء العطف
                    index += 1
                elif letters[index] == araby.ALEF_HAMZA_ABOVE and harakat[index] == araby.FATHA: # أ همزة إستفهام
                    index += 1
                else:
                    break
        else:
            while (True):
                if index >= len(letters):
                    return False , -1
                if letters[index] == araby.WAW and harakat[index] in no_harak_or_fatha: # وَ واو العطف
                    index += 1
                elif letters[index] == araby.FEH and harakat[index] in no_harak_or_fatha: #فَ فاء العطف
                    index += 1
                elif letters[index] == araby.ALEF_HAMZA_ABOVE and harakat[index] in no_harak_or_fatha: # أ همزة إستفهام
                    index += 1
                else:
                    break
        letters = letters[index:]
        harakat = harakat[index:]

        if len(letters) < 4: #فللاسم) أقل شي حرفين بعد لل)
            return False, -1 # استثناء حالة بلل (مبلول)

        next_letter_haraka = harakat[2]

        if letters[0] == araby.LAM and letters[1] == araby.LAM:
            if must_have_voweles:
                if harakat[0] in kasra_or_fatha:
                    if (harakat[1] == cls.NO_HARAKA and araby.SHADDA in next_letter_haraka) or\
                        (harakat[1] == araby.SUKUN and not araby.SHADDA in next_letter_haraka ) or\
                            (letters[2] == araby.ALEF and harakat[1] == araby.KASRA): #حالة لِلِاسْتِفَادَةِ عليها كسرة:
                        return True, index
            else:
                if harakat[0] in no_harak_or_kasra_or_fatha:
                    if harakat[1] in no_harak_or_sukun or\
                            (letters[2] == araby.ALEF and harakat[1] == araby.KASRA): #حالة لِلِاسْتِفَادَةِ عليها كسرة
                        return True, index

        return False ,-1


    @classmethod
    def is_mad_letter(cls, letter, haraka, prev_haraka) -> bool:
        return araby.is_alef(letter) or (letter == araby.WAW and haraka == cls.NO_HARAKA and araby.DAMMA in prev_haraka)\
            or(letter == araby.YEH and haraka == cls.NO_HARAKA and araby.KASRA in prev_haraka)

    @classmethod
    def is_fully_diacritized(cls, word: str, count_last_haraka: bool = True) -> bool:
        '''
        checksi whether every letter has dicaritic except for moduod letters and al alta3eef
        NOTE: this may give worng answer if there must be haraka on waw or yeh
              or the word starts with alef lam not al alta3reef
        '''
        harakat,letters = cls.extract_harakat(word)

        # check haraka on al alta3reef  and ll alta3reef
        if len(letters) > 3:
            has_al, al_index = cls.has_al_alta3reef(word, must_have_voweles=True)
            if has_al:
                if al_index + 2 <= len(harakat):  # Ensure index is within bounds
                    harakat = harakat[:al_index] + harakat[al_index + 2:]
                    letters = letters[:al_index] + letters[al_index + 2:]
            else:
                has_ll, ll_index = cls.has_ll_alta3reef(word, must_have_voweles=True)
                if has_ll:
                    if ll_index + 2 <= len(harakat):  # Ensure index is within bounds
                        harakat = harakat[:ll_index] + harakat[ll_index + 2:]
                        letters = letters[:ll_index] + letters[ll_index + 2:]

        test_range = range(len(harakat)) if count_last_haraka else range(len(harakat) - 1)

        if letters and letters[0] in [araby.WAW, araby.YEH]:
            if harakat[0] == cls.NO_HARAKA:
                return False

        for i in test_range:
            if i >= len(letters) or i >= len(harakat):
                return False  # Handle cases where the indices exceed the length of lists

            if letters[i] in cls.MODOUD_LETTERS:
                if i > 0 and cls.is_mad_letter(letters[i], harakat[i], harakat[i - 1]):
                    continue
            else:
                if harakat[i] == cls.NO_HARAKA:
                    return False
                if harakat[i] == araby.SHADDA:
                    return False

        return True

    @classmethod
    def caculate_error_on_single_sentence (cls, voweled_sentence, ground_truth_sentence,\
                gt_missing_diacritic_is_error= False):

        voweled_words = cls.split_arabic_text(voweled_sentence.strip())
        gt_words = cls.split_arabic_text(ground_truth_sentence.strip())


        word_count = 0
        letter_count = 0

        not_voweled_words_count =0

        total_wer = 0
        morph_wer = 0
        total_der = 0
        morph_der = 0

        if len(voweled_words) != len(gt_words):
            raise RuntimeError("sentences words are not the same lenght"+"[" + str(len(voweled_words)) + "] , [" + str(len(gt_words)) + "]"+" :\nsentnece 1: " + voweled_sentence  + "\nsentence 2: " + ground_truth_sentence + "\n***********************************")

        for i in range(len(voweled_words)):
            v_word = voweled_words[i].strip()
            gt_word = gt_words[i].strip()

            if v_word == "" and gt_word == "":
              continue

            if not cls.has_arabic_letter(v_word): #not an arabic word
                if not v_word == gt_word:
                    warnings.warn(f"non_arabic word is not the same  [{v_word}] , [{gt_word}]")
                #  skip
                continue

            ###### it is arabic word
            if not cls.is_fully_diacritized(v_word):
                not_voweled_words_count +=1


            v_harakat, v_letters = cls.extract_harakat(v_word)
            gt_harakat, gt_letters = cls.extract_harakat(gt_word)
            word_count +=1
            letter_count += len(gt_letters)

            if v_letters != gt_letters:
                raise RuntimeError("words don't match [" + v_word + "] , [" + gt_word + "] in sentence" + voweled_sentence)

            correct_all_but_last_letter = True
            correct_last_letter = True

            if gt_missing_diacritic_is_error or cls.is_fully_diacritized(gt_word): # check all letters
                # go through all harakat except the last one
                for j in range(len(v_harakat) - 1):
                    if v_harakat[j] !=  gt_harakat[j]:
                      total_der += 1
                      morph_der += 1
                      correct_all_but_last_letter = False
                if not correct_all_but_last_letter:
                    morph_wer +=1
                # check last letter
                if v_harakat[-1] !=  gt_harakat[-1]:
                    total_der += 1
                    correct_last_letter = False
                if not(correct_last_letter and correct_all_but_last_letter):
                    total_wer +=1
            else:
                # skip missing for missing gt diacritics if i am not missing it too

                for j in range(len(v_harakat) - 1):

                    if not v_harakat[j] == gt_harakat[j] and gt_harakat[j] == cls.NO_HARAKA:
                        letter_count -= 1
                        continue
                    if v_harakat[j] !=  gt_harakat[j]:
                      total_der += 1
                      morph_der += 1
                      correct_all_but_last_letter = False
                if not correct_all_but_last_letter:
                    morph_wer +=1
                #check last letter
                if not v_harakat[-1] == gt_harakat[-1] and gt_harakat[-1] == cls.NO_HARAKA:
                    letter_count -= 1
                    pass # last haraka is missing
                else:
                    if v_harakat[-1] !=  gt_harakat[-1]:
                      total_der += 1
                      correct_last_letter = False
                    if not(correct_last_letter and correct_all_but_last_letter):
                        total_wer +=1

        return word_count, letter_count, not_voweled_words_count, total_wer, morph_wer, total_der, morph_der


    @classmethod
    def caculate_errors_on_sentences (cls, voweled_sentences, ground_truth_sentences, gt_missing_diacritic_is_error= False) :
        if len(voweled_sentences) != len(ground_truth_sentences):
            raise RuntimeError("sentences are not the same lenght [" + str(len(voweled_sentences)) + "] , [" + str(len(ground_truth_sentences)) + "]")
        Total_WER_count = 0
        Morph_WER_count = 0
        Total_DER_count = 0
        Morph_DER_count = 0

        total_word_count = 0
        total_letter_count = 0
        total_not_voweled_words_count = 0


        for i in tqdm(range(len(voweled_sentences))):
            try:
                wc, lc, nvw, w_total_wer, w_morph_wer,w_total_der,w_morph_der = cls.caculate_error_on_single_sentence(voweled_sentences[i].strip(), ground_truth_sentences[i].strip(),\
                gt_missing_diacritic_is_error = gt_missing_diacritic_is_error)
            except RuntimeError as r:
                warnings.warn(f"Skipping example #{i}: because of {r}", RuntimeWarning)
                continue


            total_word_count += wc
            total_letter_count += lc
            total_not_voweled_words_count += nvw

            Total_WER_count += w_total_wer
            Morph_WER_count += w_morph_wer
            Total_DER_count += w_total_der
            Morph_DER_count += w_morph_der


        Total_WER = Total_WER_count/total_word_count *100
        Morph_WER = Morph_WER_count/total_word_count *100
        Total_DER = Total_DER_count/total_letter_count *100
        Morph_DER = Morph_DER_count/total_letter_count *100
        NVW = total_not_voweled_words_count/total_word_count * 100

        return Total_WER, Morph_WER, Total_DER, Morph_DER, NVW


    @classmethod
    def print_pretty_table(cls,Total_WER, Morph_WER, Total_DER, Morph_DER, NVW, gt_missing_diacritic_is_error):

        t = PrettyTable(['Count', 'Value'])
        # print("Countng 1: tagger missing is error")
        if gt_missing_diacritic_is_error:
          t.add_row(["------- Benchmark Diacritization missing is counted ERROR -------" , "--------" ])
        else:
          t.add_row(["------- Benchmark Diacritization missing is SKIPPED -------" , "--------" ])

        t.add_row(["Morpholocial DER" , "{:.4f}".format(Morph_DER)])
        t.add_row(["Total DER" , "{:.4f}".format(Total_DER)])
        t.add_row(["Morphological WER" , "{:.4f}".format(Morph_WER)])
        t.add_row(["Total WER" , "{:.4f}".format(Total_WER)])
        t.add_row(["",""])
        t.add_row(["Not Fully Diac Words Rate" , "{:.4f}".format(NVW)])
        t.add_row(["",""])
        t.add_row(["************************************","********"])

        print()
        print(t)


    @classmethod
    def report_error_on_senenteces(cls, diacritized_sentences, ground_truth_sentences, gt_missing_diacritic_is_error = False):
        """
        Calculates and prints diacritization error metrics for a list of sentences.

        This method compares predicted diacritized sentences against ground truth sentences,
        computes evaluation metrics (WER and DER), and prints the results in a formatted table.

        Args:
            diacritized_sentences (List[str]): List of predicted diacritized sentences.
            ground_truth_sentences (List[str]): List of ground truth fully diacritized sentences.
            gt_missing_diacritic_is_error (bool, optional): If True, missing diacritics in ground truth
                are treated as errors during evaluation, otherwise it is skipped. Defaults to False.

        Returns:
            None
        """
        print(f"Calculating Diacritization Errors for {len(diacritized_sentences)} sentences")
        Total_WER, Morph_WER, Total_DER, Morph_DER, NVW = ArabicDiacritizationEvaluator.caculate_errors_on_sentences(\
                            diacritized_sentences, ground_truth_sentences, gt_missing_diacritic_is_error= gt_missing_diacritic_is_error)
        cls.print_pretty_table(Total_WER, Morph_WER, Total_DER, Morph_DER, NVW, gt_missing_diacritic_is_error)



    @classmethod
    def report_errors_on_csv_file (cls, file_path, ground_truth_column_index = 0,\
                                   predicted_column_index = 1,\
                                   has_header = True,\
                                   gt_missing_diacritic_is_error = False):
        """
        Calculates diacritization error metrics from a CSV file and prints the results.

        This method reads a CSV file containing ground truth and predicted diacritized sentences,
        computes error metrics including Word Error Rate (WER) and Diacritic Error Rate (DER),
        and prints a formatted summary of the results.

        Args:
            file_path (str): Path to the input CSV file.
            ground_truth_column_index (int, optional): Index of the column containing ground truth sentences.
                Defaults to 0.
            predicted_column_index (int, optional): Index of the column containing predicted diacritized sentences.
                Defaults to 1.
            has_header (bool, optional): Indicates whether the CSV file has a header row. If True, skips the first row.
                Defaults to False.
            gt_missing_diacritic_is_error (bool, optional): Whether missing diacritics in ground truth should
                be considered errors. Defaults to False.

        Raises:
            FileNotFoundError: If the provided CSV file path does not exist.
            pd.errors.EmptyDataError: If the file exists but is empty.
            Exception: For any other unexpected errors during processing.

        Returns:
            None
        """
        try:
            if has_header:
                df = pd.read_csv(file_path, header=0)
            else:
                df = pd.read_csv(file_path, header = None)
            gt_sentences_list = df.iloc[:, ground_truth_column_index].tolist()
            diacritized_sentences_list = df.iloc[:, predicted_column_index].tolist()
            print(f"Calculating Diacritization Errors for {len(diacritized_sentences_list)} sentences")

            Total_WER, Morph_WER, Total_DER, Morph_DER, NVW = ArabicDiacritizationEvaluator.caculate_errors_on_sentences(\
                            diacritized_sentences_list, gt_sentences_list, gt_missing_diacritic_is_error= gt_missing_diacritic_is_error)
            cls.print_pretty_table(Total_WER, Morph_WER, Total_DER, Morph_DER, NVW, gt_missing_diacritic_is_error)

        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return
        except pd.errors.EmptyDataError:
            print(f"Error: File '{file_path}' is empty.")
            return
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return
