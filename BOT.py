#imports
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import discord
from discord.ext import commands

#intializing device mongodb and discord bot
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setting up intents for the Discord bot
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)
@bot.event
async def on_ready():
    print(f"Bot is ready. Logged in as {bot.user}")

# MongoDB connection setup
uri = "uri"
client: MongoClient = MongoClient(uri, server_api=ServerApi('1'))
db = client["discord_bot"]
messages_collection = db["user_messages"]

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Define the BERT model architecture
class BERTClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output  # [CLS] token
        return self.fc(pooled_output)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BERTClassifier(num_classes=2)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.to(device)
model.eval()

def predict(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs
        preds = torch.argmax(logits, dim=1)
    return preds



# Report command to handle user reports
@bot.command(name='report', help='Report a user for inappropriate behavior')
async def Report(ctx, member:discord.Member, limit:int=50):
    if member==bot.user:
        return await ctx.send("You cannot report a bot.")
    messages=list()
    async for msg in ctx.channel.history(limit=25):
        if msg.author == member:
            messages.append(msg.content)
        if len(messages) >= limit:
            break
    if not messages:
        return await ctx.send("No basis of report found.")
    else:
        predictions = predict(messages)
        toxicity ="Toxic" if (predictions == 1).sum().item()>(predictions == 0).sum().item() else "Non-Toxic"
        if toxicity=="Toxic":
            existing_user = messages_collection.find_one({"author_id": str(member.id)})
            if existing_user:
                messages_collection.update_one(
                {"author_id": str(member.id)},
                    {
                    "$set": {
                        "author_name": member.name,
                        "messages": messages
                    },
                    "$inc": {
                        "counter": 1
                    }
                }
                )
            else:
                messages_collection.insert_one({
                "author_id": str(member.id),
                "author_name": member.name,
                "messages": messages,
                "counter": 0
                })
        await ctx.send("Your report has been submitted. The user will be warned or banned based on their behavior.\n")
    counter_check = messages_collection.find_one({"author_id": str(member.id)})
    if counter_check is not None and counter_check.get('counter') == 1:
        await ctx.send(f"@{member.name} This is a warning. Please refrain from inappropriate behavior.")
    elif counter_check is not None and counter_check.get('counter') == 2:
        await member.kick(reason = "Toxicity detected after warning!")
    elif counter_check is not None and counter_check.get('counter') == 3:
        await member.ban(reason = "Repeateted Toxic behavior won't be tolerated!")

# Run the bot
bot.run('Token')

