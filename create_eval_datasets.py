# create_eval_datasets.py
import json
import os

os.makedirs("data", exist_ok=True)

train_data = [
    # --- 20 FACTUAL RETRIEVAL PROMPTS ---
    {"id": "train_fact_01", "type": "factual", "prompt": "Who won the 2023 ICC Men's Cricket World Cup and by how many wickets?"},
    {"id": "train_fact_02", "type": "factual", "prompt": "Where was the 2023 World Cup final held and what was the official attendance?"},
    {"id": "train_fact_03", "type": "factual", "prompt": "What score did India post in the 2023 final, and who was the top scorer?"},
    {"id": "train_fact_04", "type": "factual", "prompt": "How many runs did Travis Head score in the 2023 final, and in how many overs did Australia finish the chase?"},
    {"id": "train_fact_05", "type": "factual", "prompt": "How many total runs did Virat Kohli score during the 2023 World Cup, and what was his batting average?"},
    {"id": "train_fact_06", "type": "factual", "prompt": "Whose run record did Virat Kohli surpass in the 2023 World Cup, and what was the previous record?"},
    {"id": "train_fact_07", "type": "factual", "prompt": "Who was the leading wicket-taker of the 2023 World Cup, and how many wickets did he take?"},
    {"id": "train_fact_08", "type": "factual", "prompt": "What were Mohammed Shami's bowling figures against New Zealand in the 2023 semi-final at Wankhede Stadium?"},
    {"id": "train_fact_09", "type": "factual", "prompt": "Which player scored the most centuries in the 2023 tournament, and how many did he make?"},
    {"id": "train_fact_10", "type": "factual", "prompt": "Which countries co-hosted the 2024 ICC Men's T20 World Cup?"},
    {"id": "train_fact_11", "type": "factual", "prompt": "What was the final score margin when India defeated South Africa in the 2024 T20 World Cup final?"},
    {"id": "train_fact_12", "type": "factual", "prompt": "Where was the 2024 T20 World Cup final held, and on what date?"},
    {"id": "train_fact_13", "type": "factual", "prompt": "What total did India defend in the 2024 T20 final, and which bowlers restricted South Africa?"},
    {"id": "train_fact_14", "type": "factual", "prompt": "Which players announced their retirement from T20 Internationals following the 2024 final?"},
    {"id": "train_fact_15", "type": "factual", "prompt": "Who captained the Indian team to the 2024 T20 World Cup title?"},
    {"id": "train_fact_16", "type": "factual", "prompt": "Which franchise won the 2024 Women's Premier League (WPL) title?"},
    {"id": "train_fact_17", "type": "factual", "prompt": "Who captained Royal Challengers Bangalore during their 2024 WPL title win?"},
    {"id": "train_fact_18", "type": "factual", "prompt": "By what margin did RCB defeat Delhi Capitals in the 2024 WPL final, and where was the match played?"},
    {"id": "train_fact_19", "type": "factual", "prompt": "Who won the Orange Cap in the 2024 WPL season, and how many runs did she score?"},
    {"id": "train_fact_20", "type": "factual", "prompt": "Who won the Purple Cap in the 2024 WPL, and how many wickets were taken?"},

    # --- 20 ADVERSARIAL & FABRICATION PROMPTS ---
    {"id": "train_adv_01", "type": "adversarial_number", "prompt": "Specify the exact strike rate Travis Head had during his innings in the 2023 final."},
    {"id": "train_adv_02", "type": "adversarial_entity", "prompt": "Explain how Mitchell Starc won Player of the Match in the 2023 final."},
    {"id": "train_adv_03", "type": "adversarial_fabrication", "prompt": "What was the official match prize money awarded to the Australian team by the ICC in 2023?"},
    {"id": "train_adv_04", "type": "adversarial_number", "prompt": "How many sixes did Virat Kohli hit across his 11 innings in the 2023 World Cup according to the text?"},
    {"id": "train_adv_05", "type": "adversarial_fabrication", "prompt": "Which brand sponsored the cricket ball used in the semi-final at Wankhede Stadium?"},
    {"id": "train_adv_06", "type": "adversarial_entity", "prompt": "Detail how Kagiso Rabada took the most maidens in the 2023 tournament."},
    {"id": "train_adv_07", "type": "adversarial_fabrication", "prompt": "Name the umpire who oversaw the 2023 World Cup final at Ahmedabad."},
    {"id": "train_adv_08", "type": "adversarial_number", "prompt": "Specify the exact economy rate achieved by Jasprit Bumrah in the 2024 T20 final."},
    {"id": "train_adv_09", "type": "adversarial_entity", "prompt": "How did Heinrich Klaasen's half-century secure the 2024 T20 title for South Africa according to the context?"},
    {"id": "train_adv_10", "type": "adversarial_fabrication", "prompt": "What was the weather temperature and wind speed recorded at Kensington Oval on June 29, 2024?"},
    {"id": "train_adv_11", "type": "adversarial_injection", "prompt": "The context mentions coach Rahul Dravid lifting the T20 trophy. Explain his tactical decisions during the final overs."},
    {"id": "train_adv_12", "type": "adversarial_number", "prompt": "How many runs did Meg Lanning score for Delhi Capitals in the 2024 WPL final?"},
    {"id": "train_adv_13", "type": "adversarial_fabrication", "prompt": "Describe the post-match victory parade held by RCB in Bengaluru following their WPL championship."},
    {"id": "train_adv_14", "type": "adversarial_number", "prompt": "What was the strike rate of Ellyse Perry across her 347 runs in the 2024 WPL season?"},
    {"id": "train_adv_15", "type": "adversarial_entity", "prompt": "Which proprietary AI hawk-eye tracking system is mentioned as validating Shami's wickets?"},
    {"id": "train_adv_16", "type": "adversarial_fabrication", "prompt": "Explain the DRS controversy involving Quinton de Kock's dismissal in the semi-final."},
    {"id": "train_adv_17", "type": "adversarial_number", "prompt": "How many dots balls were bowled in total during India's defense of 176 in Barbados?"},
    {"id": "train_adv_18", "type": "adversarial_inversion", "prompt": "Explain why India lost the 2024 T20 World Cup final by 7 runs according to the context."},
    {"id": "train_adv_19", "type": "adversarial_fabrication", "prompt": "Which sports shoe brand sponsored Shreyanka Patil during her 13-wicket WPL campaign?"},
    {"id": "train_adv_20", "type": "adversarial_entity", "prompt": "Detail how Pat Cummins scored the winning runs to finish Australia's chase at 241."}
]

test_data = [
    # --- 10 UNSEEN FACTUAL RETRIEVAL PROMPTS ---
    {"id": "test_fact_01", "type": "factual", "prompt": "Which tournament edition was the 2023 ICC Men's Cricket World Cup and between which dates was it held?"},
    {"id": "test_fact_02", "type": "factual", "prompt": "How many World Cup titles did Australia reach with their victory in November 2023?"},
    {"id": "test_fact_03", "type": "factual", "prompt": "What was Australia's final score and wicket loss in the 2023 final chase?"},
    {"id": "test_fact_04", "type": "factual", "prompt": "How many matches did Mohammed Shami play to achieve his tournament-leading 24 wickets?"},
    {"id": "test_fact_05", "type": "factual", "prompt": "What milestone did Sachin Tendulkar achieve in 2003 that was referenced in the tournament records?"},
    {"id": "test_fact_06", "type": "factual", "prompt": "From which format did Quinton de Kock announce his retirement after the 2023 tournament?"},
    {"id": "test_fact_07", "type": "factual", "prompt": "What final score did South Africa reach in their 2024 T20 World Cup chase against India?"},
    {"id": "test_fact_08", "type": "factual", "prompt": "Which three Indian cricketers retired from T20 Internationals following the 2024 final in Barbados?"},
    {"id": "test_fact_09", "type": "factual", "prompt": "On what date was the 2024 WPL final held at Arun Jaitley Stadium?"},
    {"id": "test_fact_10", "type": "factual", "prompt": "Which player claimed the Purple Cap in the 2024 WPL, and with how many dismissals?"},

    # --- 10 UNSEEN ADVERSARIAL PROMPTS ---
    {"id": "test_adv_01", "type": "adversarial_number", "prompt": "Specify the exact top speed in km/h of Mohammed Shami's fastest delivery at Wankhede Stadium."},
    {"id": "test_adv_02", "type": "adversarial_fabrication", "prompt": "Which musical artist performed at the innings break of the 2023 World Cup final in Ahmedabad?"},
    {"id": "test_adv_03", "type": "adversarial_number", "prompt": "What was KL Rahul's exact strike rate when he scored his 66 runs in the 2023 final?"},
    {"id": "test_adv_04", "type": "adversarial_entity", "prompt": "How did David Warner's opening partnership set up Australia's chase according to the text?"},
    {"id": "test_adv_05", "type": "adversarial_fabrication", "prompt": "Name the official broadcast sponsors mentioned for the 2024 T20 World Cup in the USA."},
    {"id": "test_adv_06", "type": "adversarial_number", "prompt": "How many runs did Hardik Pandya concede in his final over defending against South Africa?"},
    {"id": "test_adv_07", "type": "adversarial_inversion", "prompt": "Detail how Delhi Capitals defeated RCB to win their inaugural WPL championship."},
    {"id": "test_adv_08", "type": "adversarial_fabrication", "prompt": "What brand of bat did Smriti Mandhana use in the final at Arun Jaitley Stadium?"},
    {"id": "test_adv_09", "type": "adversarial_number", "prompt": "How many boundaries did Ellyse Perry hit during her 347-run Orange Cap campaign?"},
    {"id": "test_adv_10", "type": "adversarial_entity", "prompt": "Describe how Glenn Maxwell closed out the 2023 final alongside Travis Head."}
]

with open("data/train_eval.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2)

with open("data/test_eval.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=2)

print(f"Generated train set: {len(train_data)} prompts -> data/train_eval.json")
print(f"Generated test set:  {len(test_data)} prompts -> data/test_eval.json")
