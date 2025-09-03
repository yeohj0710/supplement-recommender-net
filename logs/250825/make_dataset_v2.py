import argparse, csv, math, numpy as np

CATS=["psyllium_fiber","minerals","vitamin_a","iron","phosphatidylserine","folic_acid","arginine","chondroitin","coq10","collagen","vitamin_c","omega3","calcium","lutein","vitamin_d","milk_thistle","probiotics","vitamin_b","magnesium","garcinia","multivitamin","zinc"]

def sigmoid(x):
    return 1/(1+math.exp(-x))

def clamp(v,a,b):
    return a if v<a else b if v>b else v

def sample_user(rng):
    age=int(clamp(rng.normal(38,12),18,80))
    is_male=1 if rng.random()<0.5 else 0
    h_mu=175 if is_male==1 else 162
    height=float(clamp(rng.normal(h_mu,6),150,200))
    bmi=float(clamp(rng.normal(23.5+0.03*(age-38),3.2),17.0,36.0))
    weight=float(round(bmi*(height/100.0)**2,1))
    pregnant=1 if (is_male==0 and 18<=age<=45 and rng.random()<0.08) else 0
    screen_time=int(clamp(round(rng.normal(3.5,1.0)),1,5))
    eye_fatigue=int(clamp(round(clamp(rng.normal(2.5+0.5*(screen_time-3),1.0),1,5)),1,5))
    sleep_quality=int(clamp(round(rng.normal(3.0+0.2*(age>40),1.0)),1,5))
    stress_level=int(clamp(round(rng.normal(3.1,1.0)),1,5))
    digest_issue=int(clamp(round(rng.normal(2.8,1.0)),1,5))
    immune_low=int(clamp(round(rng.normal(2.9+0.2*(stress_level-3),1.0)),1,5))
    skin_issue=int(clamp(round(rng.normal(2.7,1.0)),1,5))
    hair_loss=int(clamp(round(rng.normal(2.5+0.4*(is_male==1)+0.2*(age>45),1.0)),1,5))
    joint_pain=int(clamp(round(rng.normal(2.5+0.5*(age>50),1.0)),1,5))
    bone_concern=int(clamp(round(rng.normal(2.4+0.6*(age>50),1.0)),1,5))
    lipid_issue=int(clamp(round(rng.normal(2.6+0.4*(age>45),1.0)),1,5))
    bp_issue=int(clamp(round(rng.normal(2.4+0.4*(age>45),1.0)),1,5))
    blood_sugar_issue=int(clamp(round(rng.normal(2.3+0.4*(age>45),1.0)),1,5))
    sun_low=int(clamp(round(rng.normal(3.0,1.1)),1,5))
    fish_low=int(clamp(round(rng.normal(3.0,1.1)),1,5))
    alcohol_freq=int(clamp(round(rng.normal(1.7+0.6*is_male,1.2)),0,7))
    caffeine_sensitive=1 if rng.random()<0.3 else 0
    vegan=1 if rng.random()<0.08 else 0
    low_dairy=1 if rng.random()<0.2 else 0
    activity_low=int(clamp(round(rng.normal(2.8+0.3*(age>45),1.0)),1,5))
    memory_concern=int(clamp(round(rng.normal(2.6+0.4*(stress_level-3),1.0)),1,5))
    circulation_issue=int(clamp(round(rng.normal(2.5+0.3*(age>45)+0.3*(is_male==1),1.0)),1,5))
    weight_goal=int(clamp(round(rng.normal(2.7+0.3*(bmi>=25),1.0)),1,5))
    fiber_low=int(clamp(round(rng.normal(2.8+0.4*(digest_issue>=3),1.0)),1,5))
    mineral_def=int(clamp(round(rng.normal(2.6+0.2*(vegan==1)+0.2*(low_dairy==1),1.0)),1,5))
    return {
        "f_age":age,"f_is_male":is_male,"f_height_cm":round(height,1),"f_weight_kg":weight,"f_bmi":round(bmi,1),
        "f_pregnant":pregnant,"f_screen_time":screen_time,"f_eye_fatigue":eye_fatigue,
        "f_sleep_quality":sleep_quality,"f_stress_level":stress_level,"f_digest_issue":digest_issue,"f_immune_low":immune_low,
        "f_skin_issue":skin_issue,"f_hair_loss":hair_loss,"f_joint_pain":joint_pain,"f_bone_concern":bone_concern,
        "f_lipid_issue":lipid_issue,"f_bp_issue":bp_issue,"f_blood_sugar_issue":blood_sugar_issue,
        "f_sun_exposure_low":sun_low,"f_fish_intake_low":fish_low,"f_alcohol_freq":alcohol_freq,
        "f_caffeine_sensitive":caffeine_sensitive,"f_vegan":vegan,"f_low_dairy":low_dairy,"f_activity_low":activity_low,
        "f_memory_concern":memory_concern,"f_circulation_issue":circulation_issue,"f_weight_goal":weight_goal,"f_fiber_low":fiber_low,"f_mineral_def_signs":mineral_def
    }

W={
 "psyllium_fiber":{"f_digest_issue":0.8,"f_fiber_low":0.9},
 "minerals":{"f_mineral_def_signs":0.9,"f_activity_low":0.3,"f_vegan":0.4},
 "vitamin_a":{"f_skin_issue":0.6,"f_eye_fatigue":0.3,"f_fish_intake_low":0.3},
 "iron":{"f_pregnant":1.0,"f_low_dairy":0.2},
 "phosphatidylserine":{"f_memory_concern":0.9,"f_stress_level":0.5,"f_sleep_quality":0.3},
 "folic_acid":{"f_pregnant":1.0},
 "arginine":{"f_circulation_issue":0.9,"f_is_male":0.4},
 "chondroitin":{"f_joint_pain":0.9},
 "coq10":{"f_bp_issue":0.6,"f_lipid_issue":0.5,"f_age":0.02},
 "collagen":{"f_skin_issue":0.9,"f_age":0.01},
 "vitamin_c":{"f_immune_low":0.7,"f_skin_issue":0.3},
 "omega3":{"f_fish_intake_low":0.9,"f_lipid_issue":0.7,"f_bp_issue":0.3},
 "calcium":{"f_bone_concern":0.9,"f_low_dairy":0.6,"f_age":0.02},
 "lutein":{"f_eye_fatigue":0.9,"f_screen_time":0.6,"f_age":0.01},
 "vitamin_d":{"f_sun_exposure_low":1.0,"f_bone_concern":0.4},
 "milk_thistle":{"f_alcohol_freq":0.6},
 "probiotics":{"f_digest_issue":0.9},
 "vitamin_b":{"f_stress_level":0.7,"f_sleep_quality":0.4,"f_activity_low":0.3},
 "magnesium":{"f_sleep_quality":0.6,"f_stress_level":0.5,"f_activity_low":0.3},
 "garcinia":{"f_bmi":0.5,"f_weight_goal":0.8,"f_activity_low":0.3},
 "multivitamin":{"f_activity_low":0.5,"f_vegan":0.4},
 "zinc":{"f_hair_loss":0.6,"f_immune_low":0.4,"f_is_male":0.3}
}

B={
 "psyllium_fiber":-3.2,"minerals":-3.1,"vitamin_a":-3.2,"iron":-3.3,"phosphatidylserine":-3.2,"folic_acid":-3.4,"arginine":-3.3,"chondroitin":-3.3,"coq10":-3.2,"collagen":-3.2,"vitamin_c":-3.1,"omega3":-3.1,"calcium":-3.2,"lutein":-3.1,"vitamin_d":-3.0,"milk_thistle":-3.3,"probiotics":-3.1,"vitamin_b":-3.1,"magnesium":-3.2,"garcinia":-3.2,"multivitamin":-3.1,"zinc":-3.2
}

def score_row(row, rng):
    s={}
    for c in CATS:
        v=B[c]
        for k,w in W[c].items():
            v+=w*float(row[k])
        v+=rng.normal(0,0.25)
        p=sigmoid(v*0.6)
        s[c]=p
    return s

def select_labels(scores, rng):
    items=sorted(scores.items(), key=lambda x:x[1], reverse=True)
    y={f"y_{c}":0 for c in CATS}
    taken=0
    for c,p in items:
        if p>=0.55 and taken<3:
            y[f"y_{c}"]=1
            taken+=1
    if taken==0:
        y[f"y_{items[0][0]}"]=1
    return y

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=2500)
    ap.add_argument("--outfile",type=str,default="dataset_full.csv")
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    rng=np.random.default_rng(args.seed)
    rows=[]
    for _ in range(args.n):
        r=sample_user(rng)
        s=score_row(r,rng)
        y=select_labels(s,rng)
        r.update(y)
        rows.append(r)
    feat_cols=[k for k in rows[0].keys() if k.startswith("f_")]
    label_cols=[f"y_{c}" for c in CATS]
    with open(args.outfile,"w",newline="",encoding="utf-8") as w:
        writer=csv.DictWriter(w,fieldnames=feat_cols+label_cols)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

if __name__=="__main__":
    main()
