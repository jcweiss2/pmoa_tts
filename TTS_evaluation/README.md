### To run evaluation

- Set up environment (use conda to set up a python and R environment).

- Modify the arguments (at top of the file) in compare_with_manual.R to point to the directories

- Run.

Note: It is suggested to run as a script since there are some evaluation design choices (like creating a consensus event set for apples-to-apples comparison across methods)
Note: The sentence embedding and matching is slow, so you can modify the flags in compare_with_manual.R (by (un)commenting and pointing to saved best_matches files) to load them.
```
            event.match = T,
            event.match.files = "",
            # event.match = F,
            # event.match.files = 
            #     c(
            #         "<path_to_best_matches_files_file_1>",
            #         "<path_to_best_matches_files_file_2>",
            #     ),
```
