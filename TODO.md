- ask model to provide snippets from integrate text for each find/replace, to clarify what it intended it to integrate
    - on fail, only ask the model to provide just that single failed block instead of asking it to provide all blocks again. this is only possible once the model returns what integrated text each block relates 
- only ask for start and end lines of search block instead of exact text. if there are multiple matches, ask model to provide a sufficient number of of lines at start or end to narrow down options to a single match. ensure that the search block which the verification prompt sees is not affected by this, by populating the SEARCH section of the block given in the verification prompt with the matching text from the file, instead of only including the start and end lines provided by the model


- move all files in this repo except integrate_notes.py to a new repo next to this one called zettel_inbox. still make sure to leave a copy of all non-code files in this repo, while also carrying them across.
- add a new tool, in addition to find/replace, for creating a new file. this just takes a file name to create and initial text to add. add to the instructions an explanation that if the model wants to put some of the content from the chunk into a new file and link to it, because it couldn't find anywhere else satisfactory, it should just use the file tool to create the file, and then link to the file using a wikilink [[file_name]]. importantly it must always link to any new files it creates in at least one place (add validation rule for this). enforce that it must write some new text to any new file.

- figure out how to make my notes work with butter context repo. Add symlinks automatically for all files under infofi into a dir in the butter repo. Check that this supports two way sync
- add guidance re maximum size of file
- add a copy and paste after function. also log all actions
- prevent it from adding text to index gists. Only new files
- use gemini flash