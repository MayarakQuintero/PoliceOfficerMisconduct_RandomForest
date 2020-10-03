"""Created by Sasha on June 7th
Currently some function we used in notebooks previously to add features of complaint bins, suspension length, salary
Next steps:
Use of race of victims from complaints-victims
Get number of CPD witnesses for each complaint
Get info from TRR action report maybe
"""
import pandas as pd

def relevant_complaints(complaints, complaints_accused, t1, t2):
    """Last updated by Sasha on May 31st"""
    complaints["incident_date"] = pd.to_datetime(complaints["incident_date"])
    complaints_t1 = complaints[(complaints["incident_date"].dt.year >= t1[0]) &
                               (complaints["incident_date"].dt.year <= t1[1])]
    complaints_t2 =  complaints[(complaints["incident_date"].dt.year >= t2[0]) &
                               (complaints["incident_date"].dt.year <= t2[1])]

    complaints_t1_full = pd.merge(complaints_t1[["cr_id", "incident_date"]], complaints_accused,
                                  how="left", on="cr_id")
    complaints_t2_full = pd.merge(complaints_t2[["cr_id", "incident_date"]], complaints_accused,
                                  how="left", on = "cr_id")

    return complaints_t1_full, complaints_t2_full

def complaint_bins(complaints):
    """Should take complaints_accused"""
    complaints["complaint_category"].fillna("None", inplace=True)

    complaints["complaints_binned"] = complaints["complaint_category"]
    #Fold violence with weapon down into one bin
    complaints["complaints_binned"] = complaints["complaints_binned"].map(lambda x: "Use of Force - Weapon" if
                                                                            "TASER" in x or "FIREARM" in x else x)
    #Fold non-weapon violence down into one bin
    complaints["complaints_binned"] = complaints\
                                       ["complaints_binned"].map(lambda x:
                                                                  "Use of Force - No Weapon"
                                                                  if "KICKED" in x
                                                                  or "CHOKED" in x
                                                                  or "GRAB" in x
                                                                  or "STOMP" in x
                                                                  or "ASSAULT" in x
                                                                  or "PUNCH" in x
                                                                  or "TAKE DOWN" in x
                                                                  or "UNNECESSARY PHYSICAL CONTACT" in x
                                                                  or ("EXCESSIVE FORCE" in x and
                                                                      "FIREARM" not in x and
                                                                      "TASER" not in x)
                                                                  else x)

    complaints["complaints_binned"] = complaints["complaints_binned"].map(lambda x: "Non-Violent Civilian Contact" if
                                                                            "ILLEGAL ARREST" in x or
                                                                            "DETENTION" in x or "ALTERCATION" in x or
                                                                            "WARRANT" in x else x)
    #Fold racial complaints down into one bin
    complaints["complaints_binned"] = complaints["complaints_binned"].map(lambda x:
                                                                            "Racial/Ethnic" if "RACIAL" in x else x)


    #Fold Drug use down into one bin
    complaints["complaints_binned"] = complaints["complaints_binned"].map(lambda x: "Officer Drug Use/Possession"
                                                                                      if "DRUG" in x or
                                                                                      "INTOXIC" in x or "D.U.I" in x
                                                                                      or "ALCOHOL" in x
                                                                                      else x)

    bins = ["Use of Force - Weapon", "Use of Force - No Weapon", "Racial/Ethnic", "Officer Drug Use/Possession",
            "Non-Violent Civilian Contact"]
    complaints["complaints_binned"][complaints["complaints_binned"].isin(bins) == False] = "Other/Unknown"
    
    return complaints 

def add_salary_data(salary_ranks, t1):
    salary_ranks['UID'] = salary_ranks['UID'].astype(int)
    salary_ranks['year'] = salary_ranks['year'].astype(int)
    salary_ranks_t1 = salary_ranks[(salary_ranks['year'] >= t1[0]) & (salary_ranks['year'] <= t1[1]) ]
    salary_ranks_t1_T = salary_ranks_t1.pivot(index = "UID", columns ='year', values = 'salary')
    salary_ranks_t1_T['average_salary'] = salary_ranks_t1_T.mean(axis=1, skipna=True)
    salary_ranks_t1_T_change = salary_ranks_t1_T.fillna(0)
    salary_ranks_t1_T_change['salary_change'] = salary_ranks_t1_T_change[t1[1]] - salary_ranks_t1_T_change[t1[0]]
    salary_ranks_t1_T['salary_change'] = salary_ranks_t1_T_change['salary_change']

    return salary_ranks_t1_T

def add_suspension_length(complaints):
    """Written by Sasha on May 26th"""
    final_outcome = complaints["final_outcome"].fillna("")
    complaints["suspension_length"] = final_outcome.map(lambda x: int(x.split()[0])
                                                                                if x is not None and
                                                                                "Day Suspension" in x
                                                                                else 0)

def prep_y(complaints_t2):
    '''
    Prepares target variables.
    Was moved to Feat_Engineering by Sasha on June 7th
    '''
    complaints_t2["final_outcome"] = complaints_t2["final_outcome"].str.lower()
    complaints_t2["complaint_sustained"] = \
        (complaints_t2["final_outcome"].str.contains("no action taken") == False) & \
        (complaints_t2["final_outcome"].str.contains("unknown") == False) & \
        (complaints_t2["final_outcome"].isna() == False)
    complaints_t2["target_sustained"] = complaints_t2.groupby("UID")["complaint_sustained"].transform("sum") > 0
    complaint_bins(complaints_t2) # make bins
    complaints_t2["complaints_binned"] = complaints_t2["complaints_binned"].str.lower()
    complaints_t2["comp_use_of_force"] = complaints_t2["complaints_binned"].str.contains("use of force")
    complaints_t2["target_use_of_force"] = complaints_t2.groupby("UID")["comp_use_of_force"].transform("sum") > 0
    complaints_t2["comp_racial"] = complaints_t2["complaints_binned"].str.contains("racial")
    complaints_t2["target_racial"] = complaints_t2.groupby("UID")["comp_racial"].transform("sum") > 0
    complaints_t2["comp_drug"] = complaints_t2["complaints_binned"].str.contains("drug")
    complaints_t2["target_drug"] = complaints_t2.groupby("UID")["comp_drug"].transform("sum") > 0
    complaints_t2["comp_nonviolent"] = complaints_t2["complaints_binned"].str.contains("non-violent")
    complaints_t2["target_nonviolent"] = complaints_t2.groupby("UID")["comp_nonviolent"].transform("sum") > 0
    complaints_t2["comp_other"] = complaints_t2["complaints_binned"].str.contains("other")
    complaints_t2["target_other"] = complaints_t2.groupby("UID")["comp_other"].transform("sum") > 0
    target_df = complaints_t2[["UID", "target_use_of_force", "target_drug", "target_racial",
                           "target_sustained", "target_nonviolent", "target_other"]].drop_duplicates()
    return target_df

def add_victim_race(by_officer_df, complaints_t1, complaints_victims):
    """Written on June 7th by Sasha"""
    complaints_w_victim_data = complaints_t1.merge(complaints_victims, how="left", on="cr_id")
    by_officer_groupby = pd.DataFrame(complaints_w_victim_data.groupby(["UID", "race"]).size())
    by_officer_racial_breakdown = pd.pivot_table(by_officer_groupby, values=0, index=['UID'], columns=['race']).reset_index()
    by_officer_racial_breakdown.fillna(0, inplace=True)
    by_officer_racial_breakdown.set_index("UID", inplace=True)
    by_officer_racial_breakdown.div(by_officer_racial_breakdown.sum(axis=1), axis=0)
    by_officer_racial_breakdown.columns = ["Pcnt Complaints Against " + x for x in by_officer_racial_breakdown.columns]

    return by_officer_racial_breakdown, list(by_officer_racial_breakdown.columns)

def make_target_col(final_df, desired_targets, col_name):
    '''
    Written by Lily, moved to Feat_Engineering py file by Sasha on June 7th.
    Creates a target column in the target_df that will be true if at least one of the desired targets is true.
    Inputs:
        desired_targets: list of target columns to include
        col_name: name of the new target column
    '''
    final_df[col_name] = final_df[desired_targets].any(axis='columns')
    return None

