from unittest import TestCase


import custom_rouge
import utils

pp_options = [utils.pp_option_lemmatize, utils.pp_option_stopwords, utils.pp_option_case_normalize]


def get_file_data(filename):
    text_file = open(filename, "r", encoding='utf-8')
    data = text_file.read()
    text_file.close()
    return data


def run_tests_with_data(self, test_data):
    for original, generated_one, generated_two, description in test_data:
        print(description)
        print('original: ' + original)
        rouge_one = custom_rouge.rouge_n(original, generated_one, n=1)
        rouge_two = custom_rouge.rouge_n(original, generated_two, n=1)
        print('Rouge for one: ' + str(rouge_one) + ' ' + generated_one)
        print('Rouge for two: ' + str(rouge_two) + ' ' + generated_two)
        combined = generated_two + ' ' + generated_one
        combined_rouge = custom_rouge.rouge_n(original, combined, n=1)
        print('Rouge for combined: ' + str(combined_rouge) + ' ' + combined)
        self.assertGreaterEqual(combined_rouge, rouge_one)
        self.assertGreaterEqual(combined_rouge, rouge_two)


class RougeTest(TestCase):

    def tests_from_paper(self):
        s1 = 'police killed the gunman'
        s2 = 'police kill the gunman'
        s3 = 'the gunman kill police'
        score_s2 = custom_rouge.rouge_l(s1, s2)
        self.assertEqual(score_s2, 0.75)
        score_s3 = custom_rouge.rouge_l(s1, s3)
        self.assertEqual(score_s3, 0.5)

        reference = 'affe birne club düne essen'
        summary = 'affe birne feder geld himmel. affe club insel jagd essen.'
        lcs = 4
        p = lcs / 12
        r = lcs / 5
        f = 2 * (r * p) / (r + p)
        r_p, r_r, r_f = custom_rouge.rouge_l(reference_summary=reference, created_summary=summary, pp_options=pp_options,
                                             extended_results=True)
        self.assertEqual(r_r, r)
        self.assertEqual(r_p, p)
        self.assertEqual(r_f, f)

        score_equal = custom_rouge.rouge_l(summary, summary)
        self.assertEqual(score_equal, 1)

    def test_one(self):
        original = 'Für die Frage, ob alle in Art. 6 Abs. 1 der Richtlinie 2011/83/EU genannten Informationen objektiv in einem Werbemittel dargestellt werden können, ist erheblich, welchen Anteil diese Informationen am verfügbaren Raum des vom Unternehmer ausgewählten Werbeträgers einnehmen würden; die Werbebotschaft muss gegenüber den Verbraucherinformationen nicht zurücktreten.'
        sent_1 = '(1) Zwar ist für die nach der Vorabentscheidung des Gerichtshofs der Europäischen Union maßgebliche Frage, ob alle in Art. 6 Abs. 1 der Richtlinie 2011/83/EU genannten Informationen objektiv in einem Werbemittel dargestellt werden können, erheblich, welchen Anteil diese Informationen am verfügbaren Raum des vom Unternehmer ausgewählten Werbeträgers einnehmen würden.'
        sent_2 = 'Aus der Anforderung, die Informationen objektiv in der Werbebotschaft darstellen zu können, ist zu schließen, dass die Werbebotschaft gegenüber den Verbraucherinformationen nicht zurücktreten muss.'
        rouge_v1 = custom_rouge.rouge_n(original, sent_1, 1, pp_options=[utils.pp_option_stopwords])
        rouge_v2 = custom_rouge.rouge_n(original, sent_1 + ' ' + sent_2, 1, pp_options=[utils.pp_option_stopwords])
        self.assertGreater(rouge_v2, rouge_v1)

    def test_one_match(self):
        original = 'a b c d e.'
        score = custom_rouge.rouge_n(original, 'd.', n=1)
        self.assertGreater(score, 0)

    def test_extension(self):
        original_short = 'a b c d e.'
        original_medi = 'a b c d e f g h i j k l m n o.'
        original_long = 'a b c d e f g h i j k l m n o p q r s t u v w x y z.'
        test_data = [[original_short, 'a b.', 'a b d.', 'small extension short sentence'],
                     [original_short, 'a.', 'a b c d.', 'large extension short sentence'],
                     [original_medi, 'a b c d e f g h i.', 'a b c d e f g h i j.', 'small extension medi sentence'],
                     [original_medi, 'a b c d e f g h i.', 'a b c d e f g h i m n o l.',
                      'large extension medi sentence'],
                     [original_long, 'a b c d e f g h i j k l m n o p q r s t u v.',
                      'a b c d e f g h i j k l m n o p q r s t u v w.', 'small extension long sentence'],
                     [original_long, 'a b c d e f g h i j k.',
                      'a b c d e f g h i j k l m n o p q r s t u v w.', 'large extension long sentence'],
                     ]
        print('Test extensions')
        run_tests_with_data(self, test_data)

    def test_differing(self):
        original_short = 'a b c d e.'
        original_medi = 'a b c d e f g h i j k l m n o.'
        original_long = 'a b c d e f g h i j k l m n o p q r s t u v w x y z.'

        test_data = [[original_short, 'a b c.', 'a b d.', 'small difference short sentence'],
                     [original_short, 'a e.', 'a b c.', 'large difference short sentence'],
                     [original_medi, 'a b c d e f g h i.', 'a b c d e f g h j.', 'small difference medi sentence'],
                     [original_medi, 'a b c d e f g h i.', 'a b c d j k l m.',
                      'large difference medi sentence'],
                     [original_long, 'a b c d e f g h i j k l m n o p q r s t u v.',
                      'a b c d e f g h i j k l m n o p q r s t u w.', 'small difference long sentence'],
                     [original_long, 'a b c d e f g h i j k.',
                      'a b l m n o p q r s t u v.', 'large difference long sentence'],
                     ]
        print('Test differences')
        run_tests_with_data(self, test_data)

    def test_rougel_high_precision_or_recall(self):
        gold = 'Boot.'
        created = 'Boot. Boot.'
        r_p, r_r, r_f = custom_rouge.rouge_l(created_summary=created, reference_summary=gold, extended_results=True,
                                             pp_options=[utils.pp_option_stopwords, utils.pp_option_lemmatize])
        self.assertEqual(r_p, 1 / 2)
        self.assertEqual(r_r, 1)
        self.assertEqual(r_f, 2 / 3)

        gold = 'Affe Boot. Boot Club.'
        created = 'Boot.'
        r_p, r_r, r_f = custom_rouge.rouge_l(created_summary=created, reference_summary=gold, extended_results=True,
                                             pp_options=[utils.pp_option_stopwords, utils.pp_option_lemmatize])
        self.assertEqual(r_p, 1)
        self.assertEqual(r_r, 2 / 6)
        self.assertEqual(r_f, 1 / 2)

        gold = 'Im Rahmen der bei Prüfung der Schutzschranke der Berichterstattung über Tagesereignisse gemäß § 50 ' \
               'UrhG vorzunehmenden Grundrechtsabwägung ist im Falle der Veröffentlichung eines bislang ' \
               'unveröffentlichten Werks auch das vom Urheberpersönlichkeitsrecht geschützte Interesse an einer ' \
               'Geheimhaltung des Werks zu berücksichtigen. Dieses schützt das urheberrechtsspezifische Interesse des ' \
               'Urhebers, darüber zu bestimmen, ob er mit der erstmaligen Veröffentlichung den Schritt von der ' \
               'Privatsphäre in die Öffentlichkeit tut und sich und sein Werk damit der öffentlichen Kenntnisnahme ' \
               'und Kritik aussetzt. Nicht zu berücksichtigen ist bei dieser Abwägung dagegen das Interesse an der ' \
               'Geheimhaltung von Umständen, deren Offenlegung Nachteile für die Interessen des Staates und seiner ' \
               'Einrichtungen haben könnten. Dieses Interesse ist nicht durch das Urheberpersönlichkeitsrecht, ' \
               'sondern durch andere Vorschriften - etwa das Sicherheitsüberprüfungsgesetz, § 3 Nr. 1 Buchst. b IFG ' \
               'und die strafrechtlichen Bestimmungen gegen Landesverrat und die Gefährdung der äußeren Sicherheit ' \
               'gemäß §§ 93 ff. StGB - geschützt. '
        created = 'Dieses Interesse ist vielmehr durch die allgemeinen Vorschriften - etwa das ' \
                  'Sicherheitsüberprüfungsgesetz, § 3 Nr. 1 Buchst. b IFG und die strafrechtlichen Bestimmungen gegen ' \
                  'Landesverrat und die Gefährdung der äußeren Sicherheit gemäß §§ 93 ff. '
        r_p, r_r, r_f = custom_rouge.rouge_l(created_summary=created, reference_summary=gold, extended_results=True,
                                             pp_options=[utils.pp_option_stopwords, utils.pp_option_lemmatize])
        self.assertLessEqual(r_p, 1)

        gold = 'Der Eigentümer eines Grundstücks ist hinsichtlich der von einem darauf befindlichen Baum (hier: ' \
               'Birken) ausgehenden natürlichen Immissionen auf benachbarte Grundstücke Störer i.S.d. § 1004 Abs. 1 ' \
               'BGB, wenn er sein Grundstück nicht ordnungsgemäß bewirtschaftet. Hieran fehlt es in aller Regel, ' \
               'wenn die für die Anpflanzung bestehenden landesrechtlichen Abstandsregelungen eingehalten sind. 1b. ' \
               'Ein Anspruch auf Beseitigung des Baums lässt sich in diesem Fall regelmäßig auch nicht aus dem ' \
               'nachbarlichen Gemeinschaftsverhältnis herleiten. Hält der Grundstückseigentümer die für die ' \
               'Anpflanzung bestehenden landesrechtlichen Abstandsregelungen ein, hat der Eigentümer des ' \
               'Nachbargrundstücks wegen der Beeinträchtigungen durch die von den Anpflanzungen ausgehenden ' \
               'natürlichen Immissionen weder einen Ausgleichsanspruch gemäß § 906 Abs. 2 Satz 2 BGB in unmittelbarer ' \
               'Anwendung noch einen nachbarrechtlichen Ausgleichsanspruch gemäß § 906 Abs. 2 Satz 2 analog (' \
               'Abgrenzung zu Senat, Urteil vom 27. Oktober 2017 - V ZR 8/17, ZfIR 2018, 190). '
        created = "Für die Entscheidung des Meinungsstreits ist von dem oben dargelegten Grundsatz auszugehen, " \
                  "dass der Eigentümer eines Grundstücks hinsichtlich der von einem darauf befindlichen Baum " \
                  "ausgehenden natürlichen Immissionen auf benachbarte Grundstücke Störer i.S.d. § 1004 Abs. 1 BGB " \
                  "ist, wenn er sein Grundstück nicht ordnungsgemäß bewirtschaftet. Hält der Grundstückseigentümer " \
                  "die für die Anpflanzung bestehenden landes-rechtlichen Abstandsregelungen ein, hat der Eigentümer " \
                  "des Nachbargrund-stücks wegen der Beeinträchtigungen durch die von den Anpflanzungen ausgehenden " \
                  "natürlichen Immissionen weder einen Ausgleichsanspruch gemäß § 906 Abs. 2 Satz 2 BGB in " \
                  "unmittelbarer Anwendung noch einen nachbarrechtlichen Ausgleichsanspruch gemäß § 906 Abs. 2 Satz 2 " \
                  "analog. Sind die für die Anpflanzung bestehenden landesrechtlichen Abstandsregelungen eingehalten, " \
                  "lässt sich ein Anspruch auf Beseitigung der Bäume in aller Regel - und so auch hier - nicht aus " \
                  "dem nachbarlichen Gemeinschaftsverhältnis herleiten. Gemäß § 907 Abs. 2 BGB gehören aber Bäume und " \
                  "Sträucher nicht zu den Anlagen i.S.d. § 907 Abs. 1 BGB. Ob den Grundstückseigentümer für " \
                  "natürliche Immissionen eine „Sicherungspflicht“ trifft und er damit Störer i.S.d. § 1004 Abs. 1 " \
                  "BGB ist, ist jeweils anhand der Umstände des Einzelfalls zu prüfen. Rechtsfehlerhaft ist jedoch " \
                  "die Auffassung des Berufungsgerichts, der Beklagte sei als Störer i.S.d. § 1004 Abs. 1 BGB für die " \
                  "von den Birken ausgehenden Immissionen auf das Grundstück des Klägers verantwortlich. In diesem " \
                  "Fall ist er regelmäßig schon nicht Störer, so dass es bereits an einem Beseitigungsanspruch gemäß " \
                  "§ 1004 Abs. 1 BGB fehlt und der von dem Berufungsgericht beschriebene Konflikt zwischen den Regeln " \
                  "des Bürgerlichen Gesetzbuchs und den landesrechtlichen Vorschriften nicht besteht. Voraussetzung " \
                  "hierfür ist jedoch, dass der in Anspruch genommene Grundstückseigentümer für die " \
                  "Eigentumsbeeinträchtigung verantwortlich und damit Störer i.S.d. § 1004 Abs. 1 BGB ist. "

        r_p, r_r, r_f = custom_rouge.rouge_l(created_summary=created, reference_summary=gold, extended_results=True,
                                             pp_options=[utils.pp_option_stopwords, utils.pp_option_lemmatize])
        self.assertLessEqual(r_p, 1)
        self.assertLessEqual(r_r, 1)

    def test_specific(self):
        gold = 'Diese Voraussetzungen hat der XII. Zivilsenat für den vorliegenden Fall bejaht.'
        created = 'Dies ist insbesondere der Fall, wenn die Sanktion außer Verhältnis zum Gewicht des Vertragsverstoßes und den Folgen für den Schuldner der Vertragsstrafe steht.'
        created_2 = 'Deren Untergrenze ist mit 30 € angegeben.'
        r_p, r_r, r_f = custom_rouge.rouge_l(created_summary=created, reference_summary=gold, extended_results=True,
                                             pp_options=[utils.pp_option_stopwords, utils.pp_option_lemmatize])
        r_p_2, r_r_2, r_f_2 = custom_rouge.rouge_l(created_summary=created_2, reference_summary=gold, extended_results=True,
                                                   pp_options=[utils.pp_option_stopwords, utils.pp_option_lemmatize])

        r_p_c, r_r_c, r_f_c = custom_rouge.rouge_l(created_summary=created + ' ' + created_2, reference_summary=gold,
                                                   extended_results=True,
                                                   pp_options=[utils.pp_option_stopwords, utils.pp_option_lemmatize])

        self.assertGreater(r_f, r_f_2)
        self.assertGreater(r_f_c, r_f)

    def test_switched_sentences(self):
        sent_a = 'Das ist Testsatz Nummer eins.'
        sent_b = 'Das ist Testsatz Nummer zwei.'
        gold_summary = 'Das ist ein Testsatz eins zwei.'
        _, r_ab, _ = custom_rouge.rouge_n(created_summary=sent_a + ' ' + sent_b, reference_summary=gold_summary,
                                          extended_results=True, n=1,
                                          pp_options=[utils.pp_option_stopwords, utils.pp_option_lemmatize])
        _, r_ba, _ = custom_rouge.rouge_n(created_summary=sent_b + ' ' + sent_a, reference_summary=gold_summary,
                                          extended_results=True, n=1,
                                          pp_options=[utils.pp_option_stopwords, utils.pp_option_lemmatize])
        with self.subTest():
            self.assertEqual(r_ab, r_ba, 'ROUGE 1 not equal')

        _, _, f_ab = custom_rouge.rouge_l(created_summary=sent_a + ' ' + sent_b, reference_summary=gold_summary,
                                          extended_results=True,
                                          pp_options=[utils.pp_option_stopwords, utils.pp_option_lemmatize])
        _, _, f_ba = custom_rouge.rouge_l(created_summary=sent_b + ' ' + sent_a, reference_summary=gold_summary,
                                          extended_results=True,
                                          pp_options=[utils.pp_option_stopwords, utils.pp_option_lemmatize])
        with self.subTest():
            self.assertEqual(f_ab, f_ba, 'ROUGE L not equal')

        sent_a = 'See 38 U.S.C.A. ï¿½ 5103A(d); McLendon v. Nicholson, 20 Vet. App. 79 (2006); Paralyzed Veterans of America, et. al., v. Secretary of Veterans Affairs, 345 F.3d 1334 (Fed. Cir. 2003) (holding that, if the evidence of record does not establish that the veteran suffered an event, injury, or disease in service, no reasonable possibility exists that providing a medical examination or obtaining a medical opinion would substantiate the claim); Godfrey v. Brown, 8 Vet. App. 113, 121 (1995) (holding that the Board is not required to accept a medical opinion that is based on the veteran\'s recitation of medical history); Duenas v. Principi, 18 Vet. App. 512, 519 (2004) (holding that VA is not obligated to provide an examination for a medical nexus opinion where, as here, the supporting evidence of record consists only of a lay statement).'
        sent_b = 'Although it is unquestioned that the Veteran served in Vietnam during the Vietnam War era, the primary impediment to a grant of service connection in this case is the absence of medical evidence of a current psychiatric disorder.'
        gold_summary = 'Entitlement to service connection for an acquired psychiatric disorder, to include posttraumatic stress disorder (PTSD). The Veteran had active service from April 1968 to March 1970. This matter is before the Board of Veterans\' Appeals (Board) on appeal from a rating decision in August 2007 by the above Department of Veterans Affairs (VA) Regional Office (RO). Although it is unquestioned that the Veteran served in Vietnam during the Vietnam War era, the primary impediment to a grant of service connection in this case is the absence of medical evidence of a current psychiatric disorder. While sensitivity to loud noise is the type of symptom capable of lay observation, the record is devoid of evidence to suggest that the Veteran has ever been seen or treated for PTSD or any related psychiatric condition, either during service or after service separation. Moreover as a clear preponderance of the evidence is against a finding that the Veteran has a current diagnosis of PTSD, consideration of any association between his current symptomatology and his claimed in-service stressors is not necessary. Consequently, the probative value of the Veteran\'s implied or explicit assertions that he has PTSD are greatly outweighed by the objective findings of record. Service connection for PTSD is denied.'
        _, r_ab, _ = custom_rouge.rouge_n(created_summary=sent_a + ' ' + sent_b, reference_summary=gold_summary,
                                          extended_results=True, n=1,
                                          pp_options=[utils.pp_option_stopwords, utils.pp_option_lemmatize])
        _, r_ba, _ = custom_rouge.rouge_n(created_summary=sent_b + ' ' + sent_a, reference_summary=gold_summary,
                                          extended_results=True, n=1,
                                          pp_options=[utils.pp_option_stopwords, utils.pp_option_lemmatize])
        print(r_ba)
        with self.subTest():
            self.assertEqual(r_ab, r_ba, 'ROUGE 1 not equal')

        _, _, f_ab = custom_rouge.rouge_l(created_summary=sent_a + ' ' + sent_b, reference_summary=gold_summary,
                                          extended_results=True,
                                          pp_options=[utils.pp_option_stopwords, utils.pp_option_lemmatize])
        _, _, f_ba = custom_rouge.rouge_l(created_summary=sent_b + ' ' + sent_a, reference_summary=gold_summary,
                                          extended_results=True,
                                          pp_options=[utils.pp_option_stopwords, utils.pp_option_lemmatize])
        with self.subTest():
            self.assertNotEqual(f_ab, f_ba, 'ROUGE L not equal')
