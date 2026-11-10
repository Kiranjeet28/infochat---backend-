# Golden set fields

The golden set is a UTF-8 JSON array. Each case is an object with these fields:

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Stable, unique case identifier used for comparisons between runs. |
| `question` | string | User question passed to retrieval and the answer chain. |
| `type` | string | One of `answerable`, `near_miss`, `no_answer`, `paraphrase`, `redirect`, or `adversarial`. |
| `expected_status` | string | Expected behavior: `answered`, `not_found`, or `redirect`. |
| `must_contain_facts` | list of strings | Facts that must appear in the normalized answer. Empty lists are reported as needing manual fact review. |
| `expected_notice_titles` | list of strings | Notice titles expected in retrieval and as citations. Empty for cases without a notice answer. |
| `notes` | string | Human context for the case and its expected behavior. |
| `verified` | boolean, optional | `false` marks a draft whose expected facts a person has not yet checked against the official notice. Missing means a trusted, hand-written case. The runner reports verified and unverified counts separately and unverified cases never gate the exit code. |

Run `python ingestion/validate_golden.py` to validate the file. The validator warns
when there are fewer than 50 cases or an `answerable` case has no required facts.
