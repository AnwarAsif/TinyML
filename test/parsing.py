def management_flags(report, important_date):
    '''Returns count for the number of management changes in the past 730 days
    Parameters:
    report (json): REPF reports json (returned by initialise())
    important_date (datetime): date of CLA submission or fraud detection (returned by initialise())
    Returns:
    int:change_of_manager_count
    '''
    change_of_manager_count = 0
    ActiveManagement = []
    #print(report['ORBUD_ORNNN_ID'])
    if "deputymanagement" in report['REPORT_TEXT']['reportdata']:
        if report['REPORT_TEXT']['reportdata']['deputymanagement'] is not None:
            ParticipantCapacities = report['REPORT_TEXT']['reportdata']['deputymanagement']['participantcapacities']
            if isinstance(ParticipantCapacities, dict):
                ParticipantCapacities = [ParticipantCapacities]
            if isinstance(ParticipantCapacities, list):
                for i in ParticipantCapacities:
                    if "active" in i:
                        ActiveManagement.append(i['active'])
    for i in ActiveManagement:
        if isinstance(i, dict):
            i = [i]
        for j in i:
            if "participatingsince" in j:
                if isinstance(j['participatingsince'], str):
                    j['participatingsince'] = dateutil.parser.parse(j['participatingsince'])
                diff = important_date - j['participatingsince']
                #print(diff)
                if diff.days < 730:
                    change_of_manager_count += 1
            if "complementaryparticipantcapacities" in j:
                if j['complementaryparticipantcapacities'] is not None:
                    complementary = j['complementaryparticipantcapacities']['participantcapacities']
                    if isinstance(complementary, dict):
                        complementary = [complementary]
                    for participant_cap in complementary:
                        if isinstance(participant_cap, dict):
                            participant_cap = [participant_cap]
                        for active in participant_cap:
                            if isinstance(active, dict):
                                active = [active]
                            for manager in active:
                                if "participatingsince" in manager:
                                    if isinstance(manager['participatingsince'], str):
                                        manager['participatingsince'] = dateutil.parser.parse(manager['participatingsince'])
                                    diff = important_date - manager['participatingsince']
                                    #print(diff)
                                    if diff.days < 730:
                                        change_of_manager_count += 1
    return change_of_manager_count
    #{​​​​"change_of_manager_count": change_of_manager_count}

    