"""Source registry for the fine-tuning data: where rows come from and the text written for them.

Everything here is static: Hugging Face ids, configs, splits and columns; label names; the
hand-written one-line label descriptions; criteria and level texts; instruction paraphrases;
and the per-source example counts of the fine-tune spec (96,000 train rows at scale 1.0).
`davout.train.data` turns it into examples.

Schemas were read from the real datasets on 2026-10-01 (features, label values and sample
rows); the builder re-checks every label mapping against the loaded data.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

# -- data classes -----------------------------------------------------------------------


@dataclass(frozen=True)
class Label:
    """One class of a label set.

    `raw` is the dataset's own label string, `name` a readable lower-case name, `phrase`
    what one-vs-rest templates insert for `{p}`, `desc` the one-line description and
    `group` the parent group used to draw hard negatives.
    """

    raw: str
    name: str
    phrase: str | None = None
    desc: str | None = None
    group: str | None = None
    # label-specific one-vs-rest wording (replaces the label set's templates)
    statements: tuple[str, ...] = ()
    questions: tuple[str, ...] = ()
    negations: tuple[str, ...] = ()


@dataclass(frozen=True)
class LabelSet:
    """The classes of one task and the wording used to ask about them."""

    labels: tuple[Label, ...]
    choice_instructions: tuple[str, ...] = ()
    statements: tuple[str, ...] = ()  # one-vs-rest statement templates with {p}
    questions: tuple[str, ...] = ()  # one-vs-rest question templates with {p}; end in "?"
    negations: tuple[str, ...] = ()  # negated statement templates with {p}
    true_criteria: tuple[str, ...] = ()  # "Yes" text for a label without a description
    false_criteria: tuple[str, ...] = ()  # "No" text
    quote_names: bool = False  # {p} is the styled option name, not a phrase
    hard: bool = False  # negatives come from the most similar label names; choice rows are ten-option shortlists

    def by_raw(self) -> dict[str, Label]:
        return {label.raw: label for label in self.labels}


@dataclass(frozen=True)
class TextStyle:
    """How a single-text state may be wrapped: line labels, JSON keys and the noun for templates."""

    labels: tuple[str, ...]
    keys: tuple[str, ...]
    noun: str = "text"


@dataclass(frozen=True)
class Scale:
    """One way to ask a Score question: instructions and level texts, lowest level first.

    `levels[n]` maps "terse" and "descriptive" to the n level texts; n is the native level
    count or a collapsed one (3 or 2).
    """

    instructions: tuple[str, ...]
    levels: Mapping[int, Mapping[str, tuple[str, ...]]]


@dataclass(frozen=True)
class Source:
    """One dataset (or dataset config) and how many examples of each family it contributes."""

    name: str
    number: int  # row of the spec's dataset table; 0 for sources used only in dev_xfer
    dataset: str
    config: str | None
    train_splits: tuple[str, ...]
    dev_splits: tuple[str, ...]  # held-out splits for dev_in; () = held-out rows of the train split
    columns: tuple[str, ...]
    counts: Mapping[str, int] = field(default_factory=dict)  # family -> train examples at scale 1.0
    dev_n: int = 0  # dev_in examples at scale 1.0
    revision: str | None = None
    data_files: str | None = None  # pattern with {split}; set when the repo's default loader is a script
    group: str | None = None  # sources sharing a group never reuse a row or a text


# -- label sets -------------------------------------------------------------------------

NLI_NAMES = ("entailment", "neutral", "contradiction")

YAHOO = LabelSet(
    labels=(
        Label("Society & Culture", "society and culture", "society, culture or religion",
              "religion, traditions, languages, etiquette and social issues"),
        Label("Science & Mathematics", "science and mathematics", "science or mathematics",
              "physics, chemistry, biology, astronomy, math and engineering"),
        Label("Health", "health", "health", "illness, medicine, diet, fitness and mental health"),
        Label("Education & Reference", "education and reference", "education, homework or reference material",
              "school, college, homework, studying and looking up words or facts"),
        Label("Computers & Internet", "computers and internet", "computers or the internet",
              "hardware, software, programming, websites and online services"),
        Label("Sports", "sports", "sports", "teams, athletes, matches, leagues and outdoor recreation"),
        Label("Business & Finance", "business and finance", "business, money or careers",
              "companies, jobs, investing, taxes, credit and personal finance"),
        Label("Entertainment & Music", "entertainment and music", "entertainment or music",
              "movies, television, celebrities, songs, bands and games"),
        Label("Family & Relationships", "family and relationships", "family, dating or relationships",
              "dating, marriage, friendship, parenting and family matters"),
        Label("Politics & Government", "politics and government", "politics, law or government",
              "elections, laws, the military, immigration and public policy"),
    ),
    choice_instructions=(
        "Which category does the question belong to?",
        "What is the topic of this question?",
        "Which topic best fits the post?",
        "Pick the category that best matches the question.",
        "Under which heading should this question be filed?",
    ),
    statements=("The question is about {p}.", "This post is about {p}.", "The topic of the question is {p}."),
    questions=("Is this question about {p}?", "Does the post concern {p}?", "Is the topic of this text {p}?"),
    negations=("The question is not about {p}.", "This post is not about {p}."),
    true_criteria=("the question concerns {p}",),
    false_criteria=("any other topic", "it is about something else", "the question concerns a different subject"),
)  # fmt: skip

NEWSGROUPS = LabelSet(
    labels=(
        Label("alt.atheism", "atheism", "atheism",
              "debates about atheism, belief and the existence of gods", "alt"),
        Label("comp.graphics", "computer graphics", "computer graphics",
              "image formats, rendering, 3D software and graphics programming", "comp"),
        Label("comp.os.ms-windows.misc", "ms windows", "Microsoft Windows",
              "using and troubleshooting the Microsoft Windows operating system", "comp"),
        Label("comp.sys.ibm.pc.hardware", "pc hardware", "IBM PC hardware",
              "IBM-compatible PC components such as drives, cards, motherboards and monitors", "comp"),
        Label("comp.sys.mac.hardware", "mac hardware", "Apple Macintosh hardware",
              "Apple Macintosh computers, upgrades and peripherals", "comp"),
        Label("comp.windows.x", "x window system", "the X Window System",
              "X11 programming, window managers and display servers on Unix", "comp"),
        Label("misc.forsale", "for sale", "items offered for sale",
              "classified ads offering or seeking items to buy", "misc"),
        Label("rec.autos", "cars", "cars", "automobiles, driving, car models and repairs", "rec"),
        Label("rec.motorcycles", "motorcycles", "motorcycles",
              "bikes, riding, gear and motorcycle maintenance", "rec"),
        Label("rec.sport.baseball", "baseball", "baseball",
              "baseball teams, players, games and statistics", "rec"),
        Label("rec.sport.hockey", "hockey", "ice hockey",
              "ice hockey teams, players, games and playoffs", "rec"),
        Label("sci.crypt", "cryptography", "cryptography",
              "encryption, keys, privacy and security policy", "sci"),
        Label("sci.electronics", "electronics", "electronics",
              "circuits, components and electronic devices", "sci"),
        Label("sci.med", "medicine", "medicine", "diseases, treatments, drugs and medical research", "sci"),
        Label("sci.space", "space", "space exploration",
              "spaceflight, rockets, satellites, NASA and astronomy", "sci"),
        Label("soc.religion.christian", "christianity", "Christianity",
              "Christian faith, scripture, churches and theology", "soc"),
        Label("talk.politics.guns", "gun politics", "gun politics",
              "firearms, gun control laws and the right to bear arms", "talk"),
        Label("talk.politics.mideast", "middle east politics", "Middle East politics",
              "Israel, Arab countries, Turkey, Armenia and conflicts in the region", "talk"),
        Label("talk.politics.misc", "general politics", "politics in general",
              "government policy, elections, taxes and social issues", "talk"),
        Label("talk.religion.misc", "religion", "religion in general",
              "general debates about religion, morality and beliefs", "talk"),
    ),
    choice_instructions=(
        "Which newsgroup was this post sent to?",
        "What is the post about?",
        "Which discussion group fits this message best?",
        "Which forum topic does the post belong to?",
        "Choose the subject of this post.",
    ),
    statements=("The post is about {p}.", "This message discusses {p}.", "The text deals with {p}."),
    questions=("Is this post about {p}?", "Does the message discuss {p}?", "Does the text deal with {p}?"),
    negations=("The post is not about {p}.", "This message does not discuss {p}."),
    true_criteria=("the post concerns {p}",),
    false_criteria=("any other subject", "the post is about something else", "it discusses a different topic"),
)  # fmt: skip

DBPEDIA = LabelSet(
    labels=(
        Label("Company", "company", "a company", "a business, corporation or commercial organisation"),
        Label("EducationalInstitution", "educational institution", "an educational institution",
              "a school, college or university"),
        Label("Artist", "artist", "an artist", "a musician, painter, writer, actor or other creative person"),
        Label("Athlete", "athlete", "an athlete", "a person who competes in a sport"),
        Label("OfficeHolder", "office holder", "a politician or public office holder",
              "a politician, judge or other holder of public office"),
        Label("MeanOfTransportation", "means of transportation", "a vehicle or other means of transportation",
              "a ship, aircraft, car, train or other vehicle"),
        Label("Building", "building", "a building", "a building or other man-made structure"),
        Label("NaturalPlace", "natural place", "a natural place",
              "a river, mountain, lake or other natural feature"),
        Label("Village", "village", "a village", "a village or small settlement"),
        Label("Animal", "animal", "an animal", "an animal species or group of animals"),
        Label("Plant", "plant", "a plant", "a plant species or group of plants"),
        Label("Album", "album", "a music album", "a music album or recording"),
        Label("Film", "film", "a film", "a movie"),
        Label("WrittenWork", "written work", "a written work",
              "a book, journal, newspaper or other publication"),
    ),
    choice_instructions=(
        "What kind of thing does the text describe?",
        "Which category does the subject of this article belong to?",
        "What type of entity is the article about?",
        "Classify the subject of the text.",
        "Which category fits this encyclopedia entry?",
    ),
    statements=("The text describes {p}.", "The article is about {p}.", "The subject of this entry is {p}."),
    questions=("Does the text describe {p}?", "Is this article about {p}?", "Is the subject of the text {p}?"),
    negations=("The text does not describe {p}.", "The article is not about {p}."),
    true_criteria=("the subject is {p}",),
    false_criteria=("some other kind of thing", "the subject is something else", "any other type of entity"),
)  # fmt: skip

CLINC_INTENTS: tuple[str, ...] = (
    "restaurant_reviews", "nutrition_info", "account_blocked", "oil_change_how", "time", "weather",
    "redeem_rewards", "interest_rate", "gas_type", "accept_reservations", "smart_home", "user_name",
    "report_lost_card", "repeat", "whisper_mode", "what_are_your_hobbies", "order", "jump_start",
    "schedule_meeting", "meeting_schedule", "freeze_account", "what_song", "meaning_of_life",
    "restaurant_reservation", "traffic", "make_call", "text", "bill_balance", "improve_credit_score",
    "change_language", "no", "measurement_conversion", "timer", "flip_coin", "do_you_have_pets", "balance",
    "tell_joke", "last_maintenance", "exchange_rate", "uber", "car_rental", "credit_limit", "oos",
    "shopping_list", "expiration_date", "routing", "meal_suggestion", "tire_change", "todo_list",
    "card_declined", "rewards_balance", "change_accent", "vaccines", "reminder_update", "food_last",
    "change_ai_name", "bill_due", "who_do_you_work_for", "share_location", "international_visa", "calendar",
    "translate", "carry_on", "book_flight", "insurance_change", "todo_list_update", "timezone",
    "cancel_reservation", "transactions", "credit_score", "report_fraud", "spending_history", "directions",
    "spelling", "insurance", "what_is_your_name", "reminder", "where_are_you_from", "distance", "payday",
    "flight_status", "find_phone", "greeting", "alarm", "order_status", "confirm_reservation", "cook_time",
    "damaged_card", "reset_settings", "pin_change", "replacement_card_duration", "new_card", "roll_dice",
    "income", "taxes", "date", "who_made_you", "pto_request", "tire_pressure", "how_old_are_you",
    "rollover_401k", "pto_request_status", "how_busy", "application_status", "recipe", "calendar_update",
    "play_music", "yes", "direct_deposit", "credit_limit_change", "gas", "pay_bill", "ingredients_list",
    "lost_luggage", "goodbye", "what_can_i_ask_you", "book_hotel", "are_you_a_bot", "next_song",
    "change_speed", "plug_type", "maybe", "w2", "oil_change_when", "thank_you", "shopping_list_update",
    "pto_balance", "order_checks", "travel_alert", "fun_fact", "sync_device", "schedule_maintenance", "apr",
    "transfer", "ingredient_substitution", "calories", "current_location", "international_fees",
    "calculator", "definition", "next_holiday", "update_playlist", "mpg", "min_payment",
    "change_user_name", "restaurant_suggestion", "travel_notification", "cancel", "pto_used",
    "travel_suggestion", "change_volume",
)  # fmt: skip  (ClassLabel order of clinc/clinc_oos "plus"; "oos" is index 42)

CLINC_OOS = "oos"

# The banking and credit-card domains plus two finance intents: too close to Banking77.
CLINC_EXCLUDED: frozenset[str] = frozenset({
    "transfer", "transactions", "balance", "freeze_account", "pay_bill", "bill_balance", "bill_due",
    "interest_rate", "routing", "min_payment", "order_checks", "pin_change", "report_fraud",
    "account_blocked", "spending_history", "credit_score", "report_lost_card", "credit_limit",
    "rewards_balance", "new_card", "application_status", "card_declined", "international_fees", "apr",
    "redeem_rewards", "credit_limit_change", "damaged_card", "replacement_card_duration",
    "improve_credit_score", "expiration_date", "exchange_rate", "direct_deposit",
})  # fmt: skip

CLINC = LabelSet(
    labels=tuple(
        Label(raw, raw.replace("_", " "))
        for raw in CLINC_INTENTS
        if raw != CLINC_OOS and raw not in CLINC_EXCLUDED
    ),
    choice_instructions=(
        "What is the intent of this message?",
        "Which intent best matches the user's query?",
        "What is the user asking for?",
        "Classify the query by intent.",
        "Which of these intents does the message express?",
    ),
    statements=(
        'The intent of this message is "{p}".',
        'The user\'s intent is "{p}".',
        'The query belongs to the intent "{p}".',
    ),
    questions=('Is the intent of this message "{p}"?', 'Does the query express the intent "{p}"?'),
    negations=('The intent of this message is not "{p}".', 'The user\'s intent is not "{p}".'),
    true_criteria=("the message matches this intent", "this is what the user wants"),
    false_criteria=(
        "the message expresses a different intent",
        "the user wants something else",
        "any other request",
    ),
    quote_names=True,
)

MASSIVE_INTENT_NAMES: tuple[str, ...] = (
    "alarm_query", "alarm_remove", "alarm_set", "audio_volume_down", "audio_volume_mute",
    "audio_volume_other", "audio_volume_up", "calendar_query", "calendar_remove", "calendar_set",
    "cooking_query", "cooking_recipe", "datetime_convert", "datetime_query", "email_addcontact",
    "email_query", "email_querycontact", "email_sendemail", "general_greet", "general_joke",
    "general_quirky", "iot_cleaning", "iot_coffee", "iot_hue_lightchange", "iot_hue_lightdim",
    "iot_hue_lightoff", "iot_hue_lighton", "iot_hue_lightup", "iot_wemo_off", "iot_wemo_on",
    "lists_createoradd", "lists_query", "lists_remove", "music_dislikeness", "music_likeness",
    "music_query", "music_settings", "news_query", "play_audiobook", "play_game", "play_music",
    "play_podcasts", "play_radio", "qa_currency", "qa_definition", "qa_factoid", "qa_maths", "qa_stock",
    "recommendation_events", "recommendation_locations", "recommendation_movies", "social_post",
    "social_query", "takeaway_order", "takeaway_query", "transport_query", "transport_taxi",
    "transport_ticket", "transport_traffic", "weather_query",
)  # fmt: skip

MASSIVE_INTENT = LabelSet(
    labels=tuple(
        Label(raw, raw.replace("_", " "), group=raw.split("_")[0]) for raw in MASSIVE_INTENT_NAMES
    ),
    choice_instructions=(
        "Which intent does the user's request express?",
        "What does the user want the assistant to do?",
        "Which intent label fits this command?",
        "What is the intent of the utterance?",
        "Identify the user's intent.",
        "Which action is the user asking for?",
    ),
)

MASSIVE_SCENARIO = LabelSet(
    labels=(
        Label("alarm", "alarm", "alarms", "setting, checking or removing alarms"),
        Label("audio", "audio", "the audio volume", "changing or muting the speaker volume"),
        Label("calendar", "calendar", "the calendar", "creating, checking or deleting events and reminders"),
        Label("cooking", "cooking", "cooking", "recipes and how to prepare food"),
        Label("datetime", "date and time", "the date or time",
              "the current time, dates and time zone conversion"),
        Label("email", "email", "email", "reading or sending emails and managing contacts"),
        Label("general", "general", "general chit-chat", "greetings, jokes and small talk with the assistant"),
        Label("iot", "smart home", "smart home devices",
              "lights, plugs, coffee machines and other connected devices"),
        Label("lists", "lists", "lists", "creating, reading or editing to-do and shopping lists"),
        Label("music", "music", "music information or preferences",
              "questions about songs, music likes and dislikes, and player settings"),
        Label("news", "news", "the news", "news headlines and stories"),
        Label("play", "play", "playing media", "playing music, radio, podcasts, audiobooks or games"),
        Label("qa", "question answering", "general knowledge",
              "factual questions, definitions, maths, stock prices and currency rates"),
        Label("recommendation", "recommendation", "recommendations",
              "suggestions for places, events or movies"),
        Label("social", "social media", "social media", "posting to or checking social networks"),
        Label("takeaway", "takeaway", "takeaway food", "ordering food or checking a takeaway order"),
        Label("transport", "transport", "transport", "taxis, tickets, traffic and directions"),
        Label("weather", "weather", "the weather", "forecasts and current weather conditions"),
    ),
    choice_instructions=(
        "Which domain does the request fall under?",
        "What is the user's request about?",
        "Which skill should handle this command?",
        "Which area of the assistant does the utterance concern?",
        "Classify the request by scenario.",
    ),
    statements=("The request is about {p}.", "The user is asking about {p}.", "This command concerns {p}."),
    questions=("Is this request about {p}?", "Is the user asking about {p}?", "Does this message concern {p}?"),
    negations=("The request is not about {p}.", "The user is not asking about {p}."),
    true_criteria=("the request concerns {p}",),
    false_criteria=("any other kind of request", "the request is about something else", "a different domain"),
)  # fmt: skip

TREC_COARSE = LabelSet(
    labels=(
        Label("abbreviation", "abbreviation", "an abbreviation or what one stands for",
              "abbreviations and what they stand for"),
        Label("description and abstract concepts", "description", "a description, definition or explanation",
              "definitions, explanations, reasons and how something is done"),
        Label("entities", "entity", "a thing such as an animal, product, food or event",
              "things such as animals, colors, foods, products, events and terms"),
        Label("human beings", "person", "a person or a group of people",
              "individuals, groups, titles and descriptions of people"),
        Label("locations", "location", "a place", "cities, countries, states, mountains and other places"),
        Label("numeric values", "number", "a number, date or amount",
              "counts, dates, money, distances, percentages and other quantities"),
    ),
    choice_instructions=(
        "What kind of answer does the question ask for?",
        "Which type of answer is expected?",
        "What is the question asking about?",
        "Classify the question by the type of its answer.",
        "Which answer category fits this question?",
    ),
    statements=("The question asks for {p}.", "The expected answer is {p}.", "The answer to this question is {p}."),
    questions=("Does the question ask for {p}?", "Is the expected answer {p}?"),
    negations=("The question does not ask for {p}.", "The expected answer is not {p}."),
    true_criteria=("the answer is {p}",),
    false_criteria=("the question asks for something else", "any other kind of answer"),
)  # fmt: skip

# (label_text, label_original, label_coarse_text) of SetFit/TREC-QC
TREC_FINE_NAMES: tuple[tuple[str, str, str], ...] = (
    ("a group or organization of persons", "HUM:gr", "human beings"),
    ("abbreviation", "ABBR:abb", "abbreviation"),
    ("an individual", "HUM:ind", "human beings"),
    ("animals", "ENTY:animal", "entities"),
    ("cities", "LOC:city", "locations"),
    ("colors", "ENTY:color", "entities"),
    ("countries", "LOC:country", "locations"),
    ("currency names", "ENTY:currency", "entities"),
    ("dates", "NUM:date", "numeric values"),
    ("definition of something", "DESC:def", "description and abstract concepts"),
    ("description of a person", "HUM:desc", "human beings"),
    ("description of something", "DESC:desc", "description and abstract concepts"),
    ("diseases and medicine", "ENTY:dismed", "entities"),
    ("elements and substances", "ENTY:substance", "entities"),
    ("equivalent terms", "ENTY:termeq", "entities"),
    ("events", "ENTY:event", "entities"),
    ("expression abbreviated", "ABBR:exp", "abbreviation"),
    ("food", "ENTY:food", "entities"),
    ("fractions", "NUM:perc", "numeric values"),
    ("inventions, books and other creative pieces", "ENTY:cremat", "entities"),
    ("languages", "ENTY:lang", "entities"),
    ("letters like a-z", "ENTY:letter", "entities"),
    ("linear measures", "NUM:dist", "numeric values"),
    ("manner of an action", "DESC:manner", "description and abstract concepts"),
    ("mountains", "LOC:mount", "locations"),
    ("musical instrument", "ENTY:instru", "entities"),
    ("number of something", "NUM:count", "numeric values"),
    ("organs of body", "ENTY:body", "entities"),
    ("other entities", "ENTY:other", "entities"),
    ("other locations", "LOC:other", "locations"),
    ("other numbers", "NUM:other", "numeric values"),
    ("plants", "ENTY:plant", "entities"),
    ("postcodes or other codes", "NUM:code", "numeric values"),
    ("prices", "NUM:money", "numeric values"),
    ("products", "ENTY:product", "entities"),
    ("ranks", "NUM:ord", "numeric values"),
    ("reasons", "DESC:reason", "description and abstract concepts"),
    ("religions", "ENTY:religion", "entities"),
    ("size, area and volume", "NUM:volsize", "numeric values"),
    ("speed", "NUM:speed", "numeric values"),
    ("sports", "ENTY:sport", "entities"),
    ("states", "LOC:state", "locations"),
    ("symbols and signs", "ENTY:symbol", "entities"),
    ("techniques and methods", "ENTY:techmeth", "entities"),
    ("temperature", "NUM:temp", "numeric values"),
    ("the lasting time of something", "NUM:period", "numeric values"),
    ("title of a person", "HUM:title", "human beings"),
    ("vehicles", "ENTY:veh", "entities"),
    ("weight", "NUM:weight", "numeric values"),
    ("words with a special property", "ENTY:word", "entities"),
)

TREC_FINE = LabelSet(
    labels=tuple(Label(name, name, group=coarse) for name, _orig, coarse in TREC_FINE_NAMES),
    choice_instructions=(
        "What specific kind of answer does the question ask for?",
        "Which answer type fits this question?",
        "What is the expected answer type of the question?",
        "Classify the question by its fine-grained answer type.",
    ),
    statements=(
        'The expected answer type is "{p}".',
        'The question asks for an answer of the type "{p}".',
    ),
    questions=('Is the expected answer type "{p}"?', 'Does the question ask for an answer of the type "{p}"?'),
    negations=('The expected answer type is not "{p}".',),
    true_criteria=("the answer is of this type",),
    false_criteria=("the answer is of a different type", "any other answer type"),
    quote_names=True,
)

GO_EMOTIONS = LabelSet(
    labels=(
        Label("admiration", "admiration", "admiration", "finding someone or something impressive or worthy of respect"),
        Label("amusement", "amusement", "amusement", "finding something funny or entertaining"),
        Label("anger", "anger", "anger", "strong displeasure or hostility"),
        Label("annoyance", "annoyance", "annoyance", "mild anger or irritation"),
        Label("approval", "approval", "approval", "a favourable opinion or agreement"),
        Label("caring", "caring", "care for someone", "concern and kindness towards someone"),
        Label("confusion", "confusion", "confusion", "lack of understanding or uncertainty"),
        Label("curiosity", "curiosity", "curiosity", "a desire to know or learn more"),
        Label("desire", "desire", "desire", "a strong wish for something"),
        Label("disappointment", "disappointment", "disappointment", "sadness because expectations were not met"),
        Label("disapproval", "disapproval", "disapproval", "an unfavourable opinion or disagreement"),
        Label("disgust", "disgust", "disgust", "revulsion or strong distaste"),
        Label("embarrassment", "embarrassment", "embarrassment", "shame or awkwardness"),
        Label("excitement", "excitement", "excitement", "enthusiasm and eagerness"),
        Label("fear", "fear", "fear", "being afraid or worried about a threat"),
        Label("gratitude", "gratitude", "gratitude", "thankfulness and appreciation"),
        Label("grief", "grief", "grief", "deep sorrow, especially over a loss"),
        Label("joy", "joy", "joy", "happiness or pleasure"),
        Label("love", "love", "love", "strong affection"),
        Label("nervousness", "nervousness", "nervousness", "apprehension or anxiety"),
        Label("optimism", "optimism", "optimism", "hopefulness about the future"),
        Label("pride", "pride", "pride", "satisfaction with an achievement, one's own or another's"),
        Label("realization", "realization", "a realization", "becoming aware of something"),
        Label("relief", "relief", "relief", "reassurance after anxiety or distress has passed"),
        Label("remorse", "remorse", "remorse", "regret or guilt"),
        Label("sadness", "sadness", "sadness", "emotional pain or sorrow"),
        Label("surprise", "surprise", "surprise", "astonishment at something unexpected"),
        Label(
            "neutral", "neutral", "no particular emotion", "no particular emotion",
            statements=("The comment expresses no particular emotion.", "The comment is emotionally neutral."),
            questions=("Is the comment emotionally neutral?", "Does the comment express no particular emotion?"),
            negations=("The comment is not emotionally neutral.",),
        ),
    ),
    choice_instructions=(
        "Which emotion does the comment express?",
        "What is the main emotion in this comment?",
        "How does the author feel?",
        "Which feeling best describes the tone of the text?",
        "Pick the emotion conveyed by the message.",
    ),
    statements=("The comment expresses {p}.", "The author expresses {p}.", "The text conveys {p}."),
    questions=("Does the comment express {p}?", "Is the author expressing {p}?", "Does this text convey {p}?"),
    negations=("The comment does not express {p}.", "The author is not expressing {p}."),
    true_criteria=("the comment shows {p}",),
    false_criteria=("the comment shows other feelings or none", "a different emotion, or none", "it does not"),
)  # fmt: skip

EMOTION = LabelSet(
    labels=(
        Label("sadness", "sadness", desc="feeling down, hurt or hopeless"),
        Label("joy", "joy", desc="feeling happy, content or cheerful"),
        Label("love", "love", desc="feeling affection, tenderness or longing for someone"),
        Label("anger", "anger", desc="feeling irritated, resentful or furious"),
        Label("fear", "fear", desc="feeling scared, nervous or anxious"),
        Label("surprise", "surprise", desc="feeling amazed, shocked or caught off guard"),
    ),
    choice_instructions=(
        "Which emotion does the writer express?",
        "What is the author feeling?",
        "Which feeling does this text convey?",
        "What emotion is expressed in the message?",
        "Name the emotion in this sentence.",
    ),
)

# roskoN/dailydialog act ids; id 2 is checked at build time to be the one whose utterances end in "?"
DAILYDIALOG_QUESTION_ACT = 2
DAILYDIALOG = LabelSet(
    labels=(
        Label(
            "1", "inform", desc="the speaker states a fact, an opinion or an answer",
            statements=(
                "The message gives information or states something.",
                "The speaker is telling the listener something.",
            ),
            questions=("Is the speaker providing information?", "Is this message a statement of information?"),
            negations=("The message does not give any information.",),
        ),
        Label(
            "2", "question", desc="the speaker wants an answer from the listener",
            statements=("The message asks a question.", "The speaker is asking for information."),
            questions=("Does the message ask a question?", "Is the speaker asking for information?"),
            negations=("The message does not ask a question.",),
        ),
        Label(
            "3", "directive", desc="a request, instruction, suggestion or offer",
            statements=(
                "The message makes a request, a suggestion or gives an instruction.",
                "The speaker wants the listener to do something.",
            ),
            questions=(
                "Does the message ask or tell the listener to do something?",
                "Is the speaker making a request or a suggestion?",
            ),
            negations=("The message does not ask the listener to do anything.",),
        ),
        Label(
            "4", "commissive", desc="accepting or rejecting a request or suggestion, or promising something",
            statements=(
                "The message accepts or rejects a proposal, or commits the speaker to something.",
                "The speaker agrees, refuses or promises to do something.",
            ),
            questions=(
                "Is the speaker accepting, refusing or promising something?",
                "Does the speaker commit to doing something?",
            ),
            negations=("The speaker does not agree to, refuse or promise anything.",),
        ),
    ),
    false_criteria=("the message does something else", "it is a different kind of message"),
)  # fmt: skip

HWU64_NAMES: tuple[str, ...] = (
    "alarm query", "alarm remove", "alarm set", "audio volume down", "audio volume mute", "audio volume up",
    "calendar query", "calendar remove", "calendar set", "cooking recipe", "datetime convert",
    "datetime query", "email addcontact", "email query", "email querycontact", "email sendemail",
    "general affirm", "general commandstop", "general confirm", "general dontcare", "general explain",
    "general joke", "general negate", "general praise", "general quirky", "general repeat", "iot cleaning",
    "iot coffee", "iot hue lightchange", "iot hue lightdim", "iot hue lightoff", "iot hue lighton",
    "iot hue lightup", "iot wemo off", "iot wemo on", "lists createoradd", "lists query", "lists remove",
    "music likeness", "music query", "music settings", "news query", "play audiobook", "play game",
    "play music", "play podcasts", "play radio", "qa currency", "qa definition", "qa factoid", "qa maths",
    "qa stock", "recommendation events", "recommendation locations", "recommendation movies", "social post",
    "social query", "takeaway order", "takeaway query", "transport query", "transport taxi",
    "transport ticket", "transport traffic", "weather query",
)  # fmt: skip

HWU64 = LabelSet(
    labels=tuple(Label(raw, raw, group=raw.split(" ")[0]) for raw in HWU64_NAMES),
    choice_instructions=("Which intent matches the user's message?",),
)

# coastalcph/lex_glue "ledgar": ClassLabel order of the 100 provision types (read 2026-10-02)
LEDGAR_NAMES: tuple[str, ...] = (
    "Adjustments", "Agreements", "Amendments", "Anti-Corruption Laws", "Applicable Laws", "Approvals",
    "Arbitration", "Assignments", "Assigns", "Authority", "Authorizations", "Base Salary", "Benefits",
    "Binding Effects", "Books", "Brokers", "Capitalization", "Change In Control", "Closings",
    "Compliance With Laws", "Confidentiality", "Consent To Jurisdiction", "Consents", "Construction",
    "Cooperation", "Costs", "Counterparts", "Death", "Defined Terms", "Definitions", "Disability",
    "Disclosures", "Duties", "Effective Dates", "Effectiveness", "Employment", "Enforceability",
    "Enforcements", "Entire Agreements", "Erisa", "Existence", "Expenses", "Fees", "Financial Statements",
    "Forfeitures", "Further Assurances", "General", "Governing Laws", "Headings", "Indemnifications",
    "Indemnity", "Insurances", "Integration", "Intellectual Property", "Interests", "Interpretations",
    "Jurisdictions", "Liens", "Litigations", "Miscellaneous", "Modifications", "No Conflicts",
    "No Defaults", "No Waivers", "Non-Disparagement", "Notices", "Organizations", "Participations",
    "Payments", "Positions", "Powers", "Publicity", "Qualifications", "Records", "Releases", "Remedies",
    "Representations", "Sales", "Sanctions", "Severability", "Solvency", "Specific Performance",
    "Submission To Jurisdiction", "Subsidiaries", "Successors", "Survival", "Tax Withholdings", "Taxes",
    "Terminations", "Terms", "Titles", "Transactions With Affiliates", "Use Of Proceeds", "Vacations",
    "Venues", "Vesting", "Waiver Of Jury Trials", "Waivers", "Warranties", "Withholdings",
)  # fmt: skip

LEDGAR = LabelSet(
    labels=tuple(Label(raw, raw.lower()) for raw in LEDGAR_NAMES),
    choice_instructions=(
        "Which type of contract provision is this?",
        "What is the subject of this contract clause?",
        "Which heading fits this provision best?",
        "Classify the contract clause by its topic.",
        "Under which section title would this provision appear in a contract?",
    ),
    hard=True,
)

# thunlp/few_rel "train_wiki": (Wikidata property, its name in the `names` column, one-line description)
FEWREL_RELATIONS: tuple[tuple[str, str, str], ...] = (
    ("P6", "head of government", "the object leads the government of the subject, a city, state or country"),
    ("P17", "country", "the object is the sovereign state the subject lies in or belongs to"),
    ("P22", "father", "the object is the father of the subject"),
    ("P27", "country of citizenship", "the object is the country the subject is a citizen of"),
    ("P31", "instance of", "the subject is a particular example of the class named by the object"),
    ("P39", "position held", "the object is a post or public office the subject holds or held"),
    ("P57", "director", "the object directed the subject, a film, series or play"),
    ("P58", "screenwriter", "the object wrote the script of the subject"),
    ("P84", "architect", "the object designed the subject, a building"),
    ("P86", "composer", "the object wrote the music of the subject"),
    ("P101", "field of work", "the object is the specialisation of the subject, a person or organisation"),
    ("P102", "member of political party", "the object is the political party the subject belongs to"),
    ("P105", "taxon rank", "the object is the level of the subject in the taxonomic hierarchy"),
    ("P106", "occupation", "the object is the profession of the subject"),
    ("P118", "league", "the object is the league the subject, a team or player, plays in"),
    ("P123", "publisher", "the object published the subject, a book, periodical, game or program"),
    ("P127", "owned by", "the object is the owner of the subject"),
    ("P131", "located in the administrative territorial entity",
     "the subject lies in the object, an administrative area such as a city, county or state"),
    ("P135", "movement", "the object is an artistic, literary or philosophical movement the subject belongs to"),
    ("P136", "genre", "the object is the genre of the subject, a creative work or an artist"),
    ("P137", "operator", "the object operates the subject, a facility, service or piece of equipment"),
    ("P140", "religion", "the object is the religion of the subject"),
    ("P150", "contains administrative territorial entity",
     "the object is a direct subdivision of the subject, an administrative area"),
    ("P156", "followed by", "the object comes immediately after the subject in a series"),
    ("P159", "headquarters location", "the object is the place where the subject, an organisation, has its headquarters"),
    ("P175", "performer", "the object performs the subject, a role or a musical work"),
    ("P176", "manufacturer", "the object makes the subject, a product"),
    ("P178", "developer", "the object developed the subject, such as software or a game"),
    ("P241", "military branch", "the object is the armed service the subject belongs to"),
    ("P264", "record label", "the object is the label that releases the subject's recordings"),
    ("P276", "location", "the object is the place where the subject, an object or event, is found or takes place"),
    ("P306", "operating system", "the object is the operating system the subject runs on"),
    ("P355", "subsidiary", "the object is a company or organisation controlled by the subject"),
    ("P400", "platform", "the object is the platform the subject was developed for or released on"),
    ("P403", "mouth of the watercourse", "the object is the body of water the subject, a river, drains into"),
    ("P407", "language of work or name", "the object is the language of the subject, a creative work or a name"),
    ("P449", "original network", "the object is the network that first aired the subject, a radio or television show"),
    ("P460", "said to be the same as", "the subject and the object are said to be the same thing"),
    ("P466", "occupant", "the object is a person or organisation that occupies the subject, a property"),
    ("P495", "country of origin", "the object is the country the subject, a work or product, comes from"),
    ("P527", "has part", "the object is a part of the subject"),
    ("P551", "residence", "the object is the place where the subject lives or lived"),
    ("P674", "characters", "the object is a character that appears in the subject, a work of fiction"),
    ("P706", "located on terrain feature", "the subject lies on the landform or body of water named by the object"),
    ("P710", "participant", "the object took part in the subject, an event or process"),
    ("P740", "location of formation", "the object is the place where the subject, a group or organisation, was formed"),
    ("P750", "distributor", "the object distributes the subject, a creative work"),
    ("P800", "notable work", "the object is a significant work by the subject"),
    ("P931", "place served by transport hub", "the object is the place served by the subject, an airport or station"),
    ("P937", "work location", "the object is the place where the subject, a person, was active"),
    ("P974", "tributary", "the object is a stream that flows into the subject, a river"),
    ("P991", "successful candidate", "the object is the person elected in the subject, an election"),
    ("P1001", "applies to jurisdiction",
     "the subject, an institution, law or office, has authority over the territory named by the object"),
    ("P1303", "instrument", "the object is the musical instrument the subject plays"),
    ("P1344", "participant of", "the object is an event the subject took part in"),
    ("P1346", "winner", "the object won the subject, a competition or event"),
    ("P1408", "licensed to broadcast to", "the object is the place the subject, a station, is licensed to broadcast to"),
    ("P1411", "nominated for", "the object is an award the subject was nominated for"),
    ("P1435", "heritage designation", "the object is the heritage status of the subject, a cultural or natural site"),
    ("P1877", "after a work by", "the object is the artist whose work the subject copies or is inspired by"),
    ("P1923", "participating team", "the object is a team that took part in the subject, an event"),
    ("P3373", "sibling", "the object is a brother or sister of the subject"),
    ("P3450", "sports season of league or competition", "the subject is one season of the competition named by the object"),
    ("P4552", "mountain range", "the object is the mountain range the subject belongs to"),
)  # fmt: skip

FEWREL = LabelSet(
    labels=tuple(Label(name.replace(" ", "_"), name, desc=desc) for _prop, name, desc in FEWREL_RELATIONS),
    choice_instructions=(
        "Which relation holds between the subject and the object?",
        "How is the subject related to the object according to the sentence?",
        "What is the relation between the subject and the object?",
        "Which relation does the sentence express between the subject and the object?",
        "Classify the relation that links the subject to the object.",
    ),
    hard=True,
)

FEWREL_BY_PROPERTY: dict[str, Label] = {prop: label for (prop, _n, _d), label in zip(FEWREL_RELATIONS, FEWREL.labels)}

# -- single-text wrappers ---------------------------------------------------------------

STYLES: dict[str, TextStyle] = {
    "mnli": TextStyle(("Text", "Passage", "Premise"), ("text", "passage", "premise")),
    "anli": TextStyle(("Text", "Passage", "Context"), ("text", "passage", "context")),
    "wanli": TextStyle(("Text", "Sentence", "Premise"), ("text", "sentence", "premise")),
    "vitaminc": TextStyle(("Evidence", "Text", "Passage"), ("evidence", "text", "passage")),
    "scitail": TextStyle(("Text", "Sentence", "Passage"), ("text", "sentence", "passage")),
    "squad_v2": TextStyle(("Passage", "Text", "Context"), ("passage", "text", "context"), "passage"),
    "pubmedqa": TextStyle(("Abstract", "Study", "Text"), ("abstract", "study", "text"), "abstract"),
    "qnli": TextStyle(("Sentence", "Text"), ("sentence", "text"), "sentence"),
    "tweet": TextStyle(("Tweet", "Post", "Text", "Message"), ("tweet", "post", "text"), "tweet"),
    "yahoo": TextStyle(("Post", "Text", "Message"), ("question", "post", "text"), "question"),
    "newsgroups": TextStyle(("Post", "Message", "Text", "Email"), ("post", "message", "text", "body"), "post"),
    "dbpedia": TextStyle(("Text", "Article", "Entry", "Abstract"), ("text", "article", "entry")),
    "clinc": TextStyle(
        ("Message", "Query", "User", "Request", "Customer"), ("message", "query", "text", "utterance"), "message"
    ),
    "massive": TextStyle(
        ("Message", "Request", "User", "Command", "Utterance"),
        ("message", "request", "utterance", "text", "query"),
        "request",
    ),
    "trec": TextStyle(("Query", "Text", "Input"), ("question", "query", "text"), "question"),
    "go_emotions": TextStyle(("Comment", "Text", "Reply", "Message"), ("comment", "text", "message"), "comment"),
    "emotion": TextStyle(("Text", "Message", "Post", "Tweet"), ("text", "message", "post")),
    "dailydialog": TextStyle(("Message", "Utterance", "Speaker", "Text"), ("message", "utterance", "text"), "message"),
    "amazon_reviews": TextStyle(("Review", "Text", "Customer review"), ("review", "text", "feedback"), "review"),
    "civil_comments": TextStyle(("Comment", "Post", "Text", "Reply"), ("comment", "text", "post"), "comment"),
    "ledgar": TextStyle(("Clause", "Provision", "Text", "Contract clause"), ("clause", "provision", "text"), "clause"),
}  # fmt: skip

# FewRel states name the sentence and the two entities: (JSON keys, line labels)
RELATION_FIELDS = (("sentence", "subject", "object"), ("Sentence", "Subject", "Object"))

# -- natural language inference ---------------------------------------------------------

# {h} is the hypothesis as a clause (no final full stop), {H} the hypothesis as written,
# {noun} what the state is called ("text", "passage", ...).
NLI_QUESTIONS = (
    "Is it true that {h}?",
    "Does the {noun} say that {h}?",
    "Does it follow from the {noun} that {h}?",
    "According to the {noun}, is it true that {h}?",
    "Based on the {noun}, can we conclude that {h}?",
    'Is this claim supported by the {noun}: "{H}"?',
)
NLI_FRAMED_STATEMENTS = (
    "According to the {noun}, {h}.",
    "The {noun} says that {h}.",
    "It follows from the {noun} that {h}.",
)
NLI_TRUE = (
    "the {noun} states or clearly implies it",
    "it follows from the {noun}",
    "the {noun} supports it",
)
NLI_FALSE = (
    "the {noun} contradicts it or does not say",
    "it is contradicted by the {noun} or cannot be determined from it",
    "the {noun} does not support it",
)

# -- reading ----------------------------------------------------------------------------

# Structured layouts hold the passage and the question in the state; embedded ones put the
# question ({q}) into the instruction and leave only the passage in the state.
SQUAD_STRUCT_STATEMENTS = (
    "The question can be answered from the passage.",
    "The passage contains the answer to the question.",
    "The passage answers the question.",
)
SQUAD_STRUCT_QUESTIONS = (
    "Can the question be answered from the passage?",
    "Does the passage contain the answer to the question?",
    "Is the answer to the question stated in the passage?",
)
SQUAD_EMBED_STATEMENTS = (
    'The {noun} answers the question "{q}".',
    'The {noun} contains the answer to the question "{q}".',
)
SQUAD_EMBED_QUESTIONS = (
    'Does the {noun} answer the question "{q}"?',
    'Can the question "{q}" be answered from the {noun}?',
    "Does the {noun} contain the answer to this question: {q}",  # used only when {q} ends in "?"
)
SQUAD_TRUE = ("the passage contains the answer", "the answer is stated in the passage")
SQUAD_FALSE = ("the passage does not contain the answer", "the answer cannot be found in the passage")

QNLI_STRUCT_STATEMENTS = (
    "The sentence answers the question.",
    "The sentence contains the answer to the question.",
)
QNLI_STRUCT_QUESTIONS = (
    "Does the sentence answer the question?",
    "Does the sentence contain the answer to the question?",
    "Can the question be answered using the sentence?",
)
QNLI_EMBED_STATEMENTS = (
    'The {noun} answers the question "{q}".',
    'The {noun} contains the answer to the question "{q}".',
)
QNLI_EMBED_QUESTIONS = (
    'Does the {noun} answer the question "{q}"?',
    "Does the {noun} contain the answer to this question: {q}",  # used only when {q} ends in "?"
)
QNLI_TRUE = ("the sentence contains the answer", "the answer can be read off the sentence")
QNLI_FALSE = ("the sentence does not contain the answer", "the sentence does not answer it")

PUBMED_PREFIXES = ("According to the {noun}, ", "Based on the {noun}, ")  # put before the question
PUBMED_TRUE = ("the findings support a yes answer", "yes, according to the study")
PUBMED_FALSE = ("the findings support a no answer", "no, according to the study")

# -- judgement --------------------------------------------------------------------------


@dataclass(frozen=True)
class Judgement:
    statements: tuple[str, ...]
    questions: tuple[str, ...]
    true: tuple[str, ...]
    false: tuple[str, ...]


JUDGEMENTS: dict[str, Judgement] = {
    "paws": Judgement(
        statements=(
            "The two sentences have the same meaning.",
            "The two sentences are paraphrases of each other.",
            "Both sentences say the same thing.",
        ),
        questions=(
            "Do the two sentences have the same meaning?",
            "Are the two sentences paraphrases of each other?",
            "Do both sentences say the same thing?",
        ),
        true=("they express the same meaning", "the sentences are paraphrases"),
        false=("their meanings differ", "the sentences differ in meaning even if they share most words"),
    ),
    "tweet_offensive": Judgement(
        statements=("The tweet is offensive.", "This tweet contains offensive language.", "The message is offensive."),
        questions=("Is this tweet offensive?", "Does the tweet contain offensive language?", "Is the message offensive?"),
        true=(
            "it contains insults, profanity or a targeted attack",
            "it insults, threatens or uses profane language",
        ),
        false=("it contains no insults, threats or profanity", "the language is acceptable"),
    ),
    "tweet_hate": Judgement(
        statements=(
            "The tweet is hateful towards women or immigrants.",
            "The tweet is hate speech.",
            "This tweet attacks people because of their gender or origin.",
        ),
        questions=(
            "Is this tweet hate speech?",
            "Does the tweet express hatred towards women or immigrants?",
            "Is the tweet hateful?",
        ),
        true=(
            "it attacks or demeans women or immigrants as a group",
            "it promotes hostility towards people because of their gender or origin",
        ),
        false=(
            "it is not hateful, even if it is rude or critical",
            "it does not attack people for their gender or origin",
        ),
    ),
    "tweet_irony": Judgement(
        statements=(
            "The tweet is ironic.",
            "The author is being sarcastic or ironic.",
            "This tweet is meant ironically.",
        ),
        questions=("Is this tweet ironic?", "Is the author being sarcastic?", "Is the tweet meant ironically?"),
        true=(
            "the author means the opposite of what is literally said",
            "the literal meaning is not what the author intends",
        ),
        false=("the author means what is said literally", "the tweet is sincere"),
    ),
}  # fmt: skip

PAIR_LABELS = (("Sentence 1", "Sentence 2"), ("First sentence", "Second sentence"), ("A", "B"))
PAIR_KEYS = (("sentence1", "sentence2"), ("first", "second"), ("a", "b"))

# -- score ------------------------------------------------------------------------------

# How native levels collapse: SCORE_COLLAPSE[source][n] maps a native level to a level of
# the n-level question, or None when the native level fits neither side.
SCORE_COLLAPSE: dict[str, dict[int, tuple[int | None, ...]]] = {
    "amazon_reviews": {3: (0, 0, 1, 2, 2), 2: (0, 0, None, 1, 1)},
    "stsb": {3: (0, 0, 1, 1, 2, 2), 2: (0, 0, 0, 1, 1, 1)},
    "tweet_sentiment": {2: (0, None, 1)},
    "civil_comments": {3: (0, 1, 1, 2), 2: (0, 0, None, 1)},
}
SCORE_NATIVE = {"amazon_reviews": 5, "stsb": 6, "tweet_sentiment": 3, "civil_comments": 4}

SCALES: dict[str, tuple[Scale, ...]] = {
    "amazon_reviews": (
        Scale(
            ("How satisfied is the customer with the product?", "How happy is the buyer with this purchase?"),
            {
                5: {
                    "terse": ("very dissatisfied", "dissatisfied", "neither satisfied nor dissatisfied",
                              "satisfied", "very satisfied"),
                    "descriptive": (
                        "Very dissatisfied: the product failed, broke or was unusable",
                        "Dissatisfied: more problems than benefits",
                        "Mixed: it works, with clear drawbacks",
                        "Satisfied: works well, with minor issues at most",
                        "Very satisfied: the product exceeded expectations",
                    ),
                },
                3: {
                    "terse": ("dissatisfied", "mixed feelings", "satisfied"),
                    "descriptive": (
                        "Dissatisfied: the buyer regrets the purchase",
                        "Mixed: the buyer sees both good and bad sides",
                        "Satisfied: the buyer is glad to have bought it",
                    ),
                },
                2: {
                    "terse": ("dissatisfied", "satisfied"),
                    "descriptive": (
                        "The customer is unhappy with the product",
                        "The customer is happy with the product",
                    ),
                },
            },
        ),
        Scale(
            ("How many stars does the review give?", "What star rating would the reviewer give the product?"),
            {
                5: {
                    "terse": ("1 star", "2 stars", "3 stars", "4 stars", "5 stars"),
                    "descriptive": (
                        "1 star: terrible", "2 stars: poor", "3 stars: okay", "4 stars: good", "5 stars: excellent",
                    ),
                },
                3: {
                    "terse": ("1-2 stars", "3 stars", "4-5 stars"),
                    "descriptive": (
                        "1-2 stars: a negative review",
                        "3 stars: a middling review",
                        "4-5 stars: a positive review",
                    ),
                },
                2: {
                    "terse": ("1-2 stars", "4-5 stars"),
                    "descriptive": ("Low rating (1 or 2 stars)", "High rating (4 or 5 stars)"),
                },
            },
        ),
        Scale(
            (
                "What is the sentiment of the review?",
                "How positive is the review?",
                "What is the overall tone of this product review?",
            ),
            {
                5: {
                    "terse": ("very negative", "negative", "neutral", "positive", "very positive"),
                    "descriptive": (
                        "Very negative: strong complaints and no praise",
                        "Negative: mostly complaints",
                        "Neutral: balanced or indifferent",
                        "Positive: mostly praise",
                        "Very positive: strong praise and no complaints",
                    ),
                },
                3: {
                    "terse": ("negative", "neutral", "positive"),
                    "descriptive": (
                        "Negative: the reviewer mainly complains",
                        "Neutral: balanced or indifferent",
                        "Positive: the reviewer mainly praises",
                    ),
                },
                2: {
                    "terse": ("negative", "positive"),
                    "descriptive": (
                        "Negative: the reviewer would not recommend the product",
                        "Positive: the reviewer would recommend the product",
                    ),
                },
            },
        ),
        Scale(
            ("How would the reviewer rate the product's quality?", "How good is the product according to the review?"),
            {
                5: {
                    "terse": ("awful", "bad", "average", "good", "great"),
                    "descriptive": (
                        "Awful: defective or useless",
                        "Bad: disappointing, with serious flaws",
                        "Average: does the job, nothing more",
                        "Good: solid, with small flaws",
                        "Great: works perfectly and is highly recommended",
                    ),
                },
                3: {
                    "terse": ("bad", "average", "good"),
                    "descriptive": (
                        "Bad: the product disappoints",
                        "Average: the product is acceptable",
                        "Good: the product pleases",
                    ),
                },
                2: {"terse": ("bad", "good"), "descriptive": ("Bad: not worth buying", "Good: worth buying")},
            },
        ),
    ),
    "stsb": (
        Scale(
            (
                "How similar in meaning are the two sentences?",
                "How close in meaning are the two sentences?",
                "To what degree do the two sentences mean the same thing?",
            ),
            {
                6: {
                    "terse": ("unrelated", "same topic only", "some shared details", "roughly equivalent",
                              "mostly equivalent", "fully equivalent"),
                    "descriptive": (
                        "Unrelated: the sentences are about different things",
                        "Same topic, but they say different things",
                        "Not equivalent, but they share some details",
                        "Roughly equivalent; some important information differs or is missing",
                        "Mostly equivalent; only unimportant details differ",
                        "Equivalent: they mean the same thing",
                    ),
                },
                3: {
                    "terse": ("different meaning", "partly the same meaning", "same meaning"),
                    "descriptive": (
                        "Different: at most the topic is shared",
                        "Partly the same: some details or the gist are shared",
                        "The same: equivalent apart from unimportant details",
                    ),
                },
                2: {
                    "terse": ("not equivalent", "equivalent"),
                    "descriptive": (
                        "Not equivalent: important content differs",
                        "Equivalent: the sentences convey roughly the same information",
                    ),
                },
            },
        ),
        Scale(
            ("Rate the semantic similarity of the two sentences.", "How much do the two sentences overlap in meaning?"),
            {
                6: {
                    "terse": ("none", "very low", "low", "moderate", "high", "complete"),
                    "descriptive": (
                        "None: no overlap in meaning",
                        "Very low: only the general topic is shared",
                        "Low: a few details are shared",
                        "Moderate: the gist is shared but key details differ",
                        "High: only minor details differ",
                        "Complete: identical in meaning",
                    ),
                },
                3: {
                    "terse": ("low", "medium", "high"),
                    "descriptive": (
                        "Low: little or no shared meaning",
                        "Medium: partly shared meaning",
                        "High: nearly or fully the same meaning",
                    ),
                },
                2: {"terse": ("low", "high"), "descriptive": ("Low similarity", "High similarity")},
            },
        ),
    ),
    "tweet_sentiment": (
        Scale(
            (
                "What is the sentiment of the tweet?",
                "What is the overall tone of this tweet?",
                "How does the author feel about what they are discussing?",
            ),
            {
                3: {
                    "terse": ("negative", "neutral", "positive"),
                    "descriptive": (
                        "Negative: the author is unhappy, critical or upset",
                        "Neutral: factual, mixed or without clear feeling",
                        "Positive: the author is pleased, approving or excited",
                    ),
                },
                2: {
                    "terse": ("negative", "positive"),
                    "descriptive": ("Negative: unhappy or critical", "Positive: pleased or approving"),
                },
            },
        ),
        Scale(
            ("How favourable is the author's attitude?", "How positive is the tweet?"),
            {
                3: {
                    "terse": ("unfavourable", "neither", "favourable"),
                    "descriptive": (
                        "Unfavourable: a complaint, criticism or bad news",
                        "Neither: plain information or mixed feelings",
                        "Favourable: praise, excitement or good news",
                    ),
                },
                2: {
                    "terse": ("unfavourable", "favourable"),
                    "descriptive": (
                        "Unfavourable: a complaint, criticism or bad news",
                        "Favourable: praise, excitement or good news",
                    ),
                },
            },
        ),
    ),
    "civil_comments": (
        Scale(
            ("How toxic is the comment?", "How toxic is this comment likely to seem to readers?"),
            {
                4: {
                    "terse": ("not toxic", "slightly toxic", "toxic", "very toxic"),
                    "descriptive": (
                        "Not toxic: a civil comment",
                        "Slightly toxic: a few readers might find it rude",
                        "Toxic: many readers would find it rude or disrespectful",
                        "Very toxic: nearly all readers would find it hateful, insulting or aggressive",
                    ),
                },
                3: {
                    "terse": ("not toxic", "borderline", "clearly toxic"),
                    "descriptive": (
                        "Not toxic: nobody would object",
                        "Borderline: some readers would object",
                        "Clearly toxic: almost everyone would object",
                    ),
                },
                2: {
                    "terse": ("not toxic", "toxic"),
                    "descriptive": (
                        "Acceptable: civil or only mildly rude",
                        "Toxic: hateful, insulting or aggressive",
                    ),
                },
            },
        ),
        Scale(
            ("How rude or disrespectful is the comment?", "How disrespectful is this comment towards others?"),
            {
                4: {
                    "terse": ("not at all", "a little", "quite", "extremely"),
                    "descriptive": (
                        "Not at all: polite or neutral",
                        "A little: a mildly rude remark",
                        "Quite: disrespectful or insulting",
                        "Extremely: abusive, hateful or threatening",
                    ),
                },
                3: {
                    "terse": ("not at all", "somewhat", "extremely"),
                    "descriptive": (
                        "Not at all: polite or neutral",
                        "Somewhat: part of the audience would take offence",
                        "Extremely: abusive, hateful or threatening",
                    ),
                },
                2: {
                    "terse": ("not rude", "rude"),
                    "descriptive": ("Not rude: respectful or neutral", "Rude: insulting, hateful or aggressive"),
                },
            },
        ),
    ),
}  # fmt: skip

# civil_comments toxicity is the share of annotators who found the comment toxic.
CIVIL_BINS = (0.0, 0.3, 0.7)  # level 0: == 0; 1: <= 0.3; 2: <= 0.7; 3: above

# -- "none of the above" ----------------------------------------------------------------

OTHER_NAMES = ("other", "other", "other", "none of the above", "none of these", "something else")
OTHER_DESCS = (
    "none of the listed options applies",
    "anything that does not fit the other options",
    "the text fits none of the named options",
)

# -- dev_xfer canonical formats ---------------------------------------------------------

XFER_MULTIRC_INSTRUCTIONS = "According to the passage, is `answer` a correct answer to `question`?"
XFER_APP_REVIEWS_INSTRUCTIONS = "How satisfied is the user with the app?"
XFER_APP_REVIEWS_LEVELS = ("Very dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very satisfied")

# -- registry ---------------------------------------------------------------------------

SOURCES: dict[str, Source] = {
    s.name: s
    for s in (
        Source("mnli", 1, "nyu-mll/multi_nli", "default", ("train",), ("validation_matched",),
               ("premise", "hypothesis", "label"), {"nli": 10_000}, 84),
        Source("anli", 2, "facebook/anli", "plain_text", ("train_r1", "train_r2", "train_r3"),
               ("dev_r1", "dev_r2", "dev_r3"), ("premise", "hypothesis", "label"), {"nli": 6_000}, 84),
        Source("wanli", 3, "alisawuffles/WANLI", "default", ("train",), ("test",),
               ("premise", "hypothesis", "gold"), {"nli": 3_000}, 83),
        Source("vitaminc", 4, "tals/vitaminc", "default", ("train",), ("validation",),
               ("evidence", "claim", "label"), {"nli": 3_000}, 83),
        Source("scitail", 5, "allenai/scitail", "tsv_format", ("train",), ("validation",),
               ("premise", "hypothesis", "label"), {"nli": 2_000}, 83),
        Source("squad_v2", 6, "rajpurkar/squad_v2", "squad_v2", ("train",), ("validation",),
               ("context", "question", "answers"), {"reading": 3_500}, 83),
        Source("pubmedqa", 7, "qiaojin/PubMedQA", "pqa_artificial", ("train",), (),
               ("question", "context", "final_decision"), {"reading": 2_500}, 83),
        Source("qnli", 8, "nyu-mll/glue", "qnli", ("train",), ("validation",),
               ("question", "sentence", "label"), {"reading": 2_000}, 83),
        Source("paws", 9, "google-research-datasets/paws", "labeled_final", ("train",), ("validation",),
               ("sentence1", "sentence2", "label"), {"judgement": 2_000}, 83),
        Source("tweet_offensive", 10, "cardiffnlp/tweet_eval", "offensive", ("train",), ("validation",),
               ("text", "label"), {"judgement": 667}, 28),
        Source("tweet_hate", 10, "cardiffnlp/tweet_eval", "hate", ("train",), ("validation",),
               ("text", "label"), {"judgement": 667}, 28),
        Source("tweet_irony", 10, "cardiffnlp/tweet_eval", "irony", ("train",), ("validation",),
               ("text", "label"), {"judgement": 666}, 27),
        Source("yahoo", 11, "community-datasets/yahoo_answers_topics", "yahoo_answers_topics", ("train",),
               ("test",), ("question_title", "question_content", "topic"),
               {"choice": 4_000, "ovr": 3_000, "candidate": 500}, 84),
        Source("newsgroups", 12, "SetFit/20_newsgroups", "default", ("train",), ("test",),
               ("text", "label_text"), {"choice": 3_000, "ovr": 2_500}, 84),
        Source("dbpedia", 13, "fancyzhx/dbpedia_14", "dbpedia_14", ("train",), ("test",),
               ("title", "content", "label"), {"choice": 3_000, "ovr": 1_500, "candidate": 500}, 84),
        Source("clinc", 14, "clinc/clinc_oos", "plus", ("train",), ("validation",),
               ("text", "intent"), {"choice": 6_000, "ovr": 1_500, "candidate": 2_500}, 84),
        Source("massive_intent", 15, "SetFit/amazon_massive_intent_en-US", "default", ("train",),
               ("validation",), ("id", "text", "label_text"), {"choice": 4_000, "candidate": 2_500}, 84,
               group="massive"),
        Source("massive_scenario", 16, "SetFit/amazon_massive_scenario_en-US", "default", ("train",),
               ("validation",), ("id", "text", "label_text"), {"choice": 2_000, "ovr": 1_000}, 83,
               group="massive"),
        Source("trec", 17, "SetFit/TREC-QC", "default", ("train",), ("test",),
               ("text", "label_coarse_text", "label_text"), {"choice": 2_000, "ovr": 1_000}, 83),
        Source("go_emotions", 18, "google-research-datasets/go_emotions", "simplified", ("train",),
               ("validation",), ("text", "labels"), {"choice": 2_000, "ovr": 1_500}, 83),
        Source("emotion", 19, "dair-ai/emotion", "split", ("train",), ("validation",),
               ("text", "label"), {"choice": 2_000}, 83),
        # The repo's default loader is a script; its parquet conversion branch holds the same rows.
        Source("dailydialog", 20, "roskoN/dailydialog", "full", ("train",), ("validation",),
               ("utterances", "acts"), {"ovr": 2_000}, 83,
               revision="refs/convert/parquet", data_files="full/{split}/*.parquet"),
        Source("amazon_reviews", 21, "SetFit/amazon_reviews_multi_en", "default", ("train",),
               ("validation",), ("text", "label"), {"score": 6_000}, 84),
        Source("stsb", 22, "sentence-transformers/stsb", "default", ("train",), ("validation",),
               ("sentence1", "sentence2", "score"), {"score": 3_000}, 83),
        Source("tweet_sentiment", 23, "cardiffnlp/tweet_eval", "sentiment", ("train",), ("validation",),
               ("text", "label"), {"score": 1_500}, 83),
        Source("civil_comments", 24, "google/civil_comments", "default", ("train",), ("validation",),
               ("text", "toxicity"), {"score": 1_500}, 83),
    )
}  # fmt: skip

# The candidate-stage experiment ("cand_v1"): two many-class sets, train splits only, turned into
# candidate rows and ten-option shortlists with hard negatives. Never part of the default mix.
CAND_SOURCES: dict[str, Source] = {
    s.name: s
    for s in (
        Source("ledgar", 25, "coastalcph/lex_glue", "ledgar", ("train",), (), ("text", "label"),
               {"candidate": 7_000, "choice": 3_000}, 100),
        # The repo's default loader is a script; its parquet conversion branch holds the same rows.
        Source("fewrel", 26, "thunlp/few_rel", "default", ("train_wiki",), (),
               ("relation", "tokens", "head", "tail", "names"), {"candidate": 7_000, "choice": 3_000}, 100,
               revision="refs/convert/parquet", data_files="default/{split}/*.parquet"),
    )
}  # fmt: skip



def _rescaled(sources: Mapping[str, Source], total: int) -> dict[str, Source]:
    """The same sources with their example counts scaled to sum to `total` (largest remainder),
    so every (source, family) cell keeps its share; they contribute no dev_in rows."""
    cells = [(s.name, family, n) for s in sources.values() for family, n in s.counts.items()]
    whole = sum(n for _name, _family, n in cells)
    exact = [total * n / whole for _name, _family, n in cells]
    counts = [int(x) for x in exact]
    for i in sorted(range(len(cells)), key=lambda i: (-(exact[i] - counts[i]), i))[: total - sum(counts)]:
        counts[i] += 1
    scaled: dict[str, dict[str, int]] = {}
    for (name, family, _n), n in zip(cells, counts):
        scaled.setdefault(name, {})[family] = n
    return {name: replace(src, counts=scaled[name], dev_n=0) for name, src in sources.items()}


# "cand_mix_v1": the candidate-stage rows plus as many rows of the default mix, in its proportions.
CAND_MIX_SOURCES: dict[str, Source] = {**CAND_SOURCES, **_rescaled(SOURCES, 20_000)}

# Named training mixes for `build(recipe=...)`; "v1" is the fine-tune spec's 96,000 rows.
RECIPES: dict[str, dict[str, Source]] = {"v1": SOURCES, "cand_v1": CAND_SOURCES, "cand_mix_v1": CAND_MIX_SOURCES}

# Sources never trained on; dev_xfer rows come from these, one canonical format each.
XFER_SOURCES: dict[str, Source] = {
    s.name: s
    for s in (
        Source("hwu64", 0, "FastFit/hwu_64", "default", (), ("test",), ("text", "label"),
               {"choice": 150, "candidate": 150}),
        Source("rte", 0, "aps/super_glue", "rte", (), ("validation",), ("premise", "hypothesis", "label"),
               {"nli": 277}),
        # 83 distinct validation paragraphs allow 166 rows at two per paragraph; the rest comes
        # from the train split, which is not used for training either.
        Source("multirc", 0, "aps/super_glue", "multirc", (), ("validation", "train"),
               ("paragraph", "question", "answer", "label"), {"reading": 250}),
        Source("app_reviews", 0, "sealuzh/app_reviews", "default", (), ("train",), ("review", "star"),
               {"score": 250}),
    )
}  # fmt: skip

# Fallback of the spec: if dailydialog's "question" act cannot be confirmed, its examples go here.
DAILYDIALOG_FALLBACK = ("yahoo", "ovr")

# Never used for training: the six benchmark datasets and their close relatives.
EXCLUDED_DATASETS: frozenset[str] = frozenset({
    "google/boolq", "ucirvine/sms_spam", "stanfordnlp/sst2", "fancyzhx/ag_news",
    "legacy-datasets/banking77", "PolyAI/banking77", "Yelp/yelp_review_full",
    "SetFit/sst5", "stanfordnlp/imdb", "cornell-movie-review-data/rotten_tomatoes",
    "fancyzhx/amazon_polarity", "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
})  # fmt: skip
EXCLUDED_CONFIGS: frozenset[tuple[str, str]] = frozenset({("aps/super_glue", "boolq"), ("nyu-mll/glue", "sst2")})

# Wrapper of the private routing eval; training states must never contain it.
FORBIDDEN_WRAPPER = "The user wrote this message:"

# Held-out task -> training sources (spec table numbers) that are close to it.
NEAR_DOMAIN: dict[str, dict[str, object]] = {
    "ag_news": {"sources": [11, 12], "note": "topic names overlap"},
    "yelp": {"sources": [21, 23], "note": "#21 is the same task on another platform"},
    "sst2": {"sources": [21, 23], "note": "no binary-sentiment Noul is trained"},
    "boolq": {"sources": [6, 8, 7], "note": "#6 and #8 use Wikipedia passages"},
    "banking77": {"sources": [14, 15], "note": "task type only, after the CLINC exclusions"},
    "sms_spam": {"sources": [], "note": "none"},
    "routing_eval": {"sources": [20, 11, 12, 13, 14, 16, 17, 18], "note": "#20 and topic one-vs-rest are closest in form"},
    "dev_xfer/hwu64": {
        "sources": [15, 16],
        "note": "HWU64 and MASSIVE share utterances and the intent taxonomy; HWU rows whose text "
        "occurs in MASSIVE or CLINC are removed, but the label space is still near-domain",
    },
}  # fmt: skip


def label_sets(source: str) -> dict[str, LabelSet]:
    """Label sets of a classification source, keyed by the name used in example metadata."""
    return {
        "yahoo": {"topic": YAHOO},
        "newsgroups": {"group": NEWSGROUPS},
        "dbpedia": {"type": DBPEDIA},
        "clinc": {"intent": CLINC},
        "massive_intent": {"intent": MASSIVE_INTENT},
        "massive_scenario": {"scenario": MASSIVE_SCENARIO},
        "trec": {"coarse": TREC_COARSE, "fine": TREC_FINE},
        "go_emotions": {"emotion": GO_EMOTIONS},
        "emotion": {"emotion": EMOTION},
        "dailydialog": {"act": DAILYDIALOG},
        "hwu64": {"intent": HWU64},
        "ledgar": {"provision": LEDGAR},
        "fewrel": {"relation": FEWREL},
    }.get(source, {})
