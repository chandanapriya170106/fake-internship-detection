"""Creates a SYNTHETIC demo dataset so the project runs out of the box.

For real results, download the Kaggle 'Real / Fake Job Postings' (EMSCAD)
dataset and save it as data/fake_job_postings.csv (column `fraudulent` = 0/1).
"""
import random
import pandas as pd

random.seed(42)
ROLES = ["Data Analyst", "Python Developer", "Web Development", "Machine Learning", "Marketing",
         "Content Writing", "Graphic Design", "Cyber Security", "Android Development", "HR"]
COMPANIES = ["Infosys", "TCS Digital Labs", "Zoho Corp", "Flipkart", "Razorpay", "Freshworks",
             "Swiggy", "Tech Mahindra", "Wipro", "Paytm"]

REAL_DESC = [
    "Selected candidates will work with our {r} team on live projects under a mentor.",
    "The internship includes an online test and two interview rounds. Duration is {m} months.",
    "You will assist engineers, attend weekly reviews and present your work at the end.",
    "Applicants must be pursuing a B.Tech or MCA degree with basic knowledge of the field.",
]
REAL_BEN = ["Certificate on completion", "Stipend of Rs {s} per month", "Pre-placement offer for top performers", "Flexible hours"]
FAKE_DESC = [
    "URGENT hiring!!! Direct selection, no interview needed. Apply immediately, limited seats.",
    "Earn Rs {s} per month working from home with only 2 hours daily. 100% selection guaranteed.",
    "Pay a registration fee of Rs 999 and a refundable security deposit to confirm your offer letter.",
    "Contact our HR on WhatsApp or Telegram now. Send your Aadhaar and bank details to get started.",
    "Instant offer letter within 24 hours. Training fee of Rs 1500 for kit and certificate.",
]
FAKE_EMAIL = ["hr.recruit{n}@gmail.com", "careers.{n}@yahoo.com", "jobs{n}@outlook.com"]


def real_post():
    r = random.choice(ROLES); c = random.choice(COMPANIES)
    desc = " ".join(random.sample(REAL_DESC, 3)).format(r=r, m=random.choice([2, 3, 6]))
    return dict(
        title=f"{r} Intern",
        company_profile=f"{c} is an established company with offices across India. Visit https://www.{c.split()[0].lower()}.com",
        description=desc,
        requirements="Good communication, basic programming, willingness to learn. Apply via careers portal.",
        benefits=random.choice(REAL_BEN).format(s=random.choice([8000, 10000, 15000, 20000])),
        fraudulent=0)


def fake_post():
    r = random.choice(ROLES)
    desc = " ".join(random.sample(FAKE_DESC, random.randint(2, 4))).format(s=random.choice([40000, 60000, 80000]))
    has_name = random.random() < 0.3
    return dict(
        title=f"{r} Internship - Work From Home!!!",
        company_profile=("Global Solutions Pvt Ltd" if has_name else ""),
        description=desc,
        requirements="Anyone can apply. No skills needed. Mail resume to " + random.choice(FAKE_EMAIL).format(n=random.randint(1, 99)),
        benefits="Huge stipend, certificate, guaranteed job",
        fraudulent=1)


rows = [real_post() for _ in range(450)] + [fake_post() for _ in range(150)]
random.shuffle(rows)
pd.DataFrame(rows).to_csv("data/sample_postings.csv", index=False)
print("Saved data/sample_postings.csv with", len(rows), "rows")
