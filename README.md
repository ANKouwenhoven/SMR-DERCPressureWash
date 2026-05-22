<html>
<body>
	<h1>Git instructions for neatly managing branches:</h1>
	<p>Check for changes within your own branch:</p>
	<blockquote>
		<p>- git fetch</p>
	</blockquote>
	<p>If it tells you there are changes, import the changes:</p>
	<blockquote>
		<p>- git pull</p>
	</blockquote>
	<p>Navigating to a branch (type without &lt;&lt; &gt;&gt;):</p>
	<blockquote>
		<p>- git checkout &lt;&lt;branch_name&gt;&gt;</p>
	</blockquote>
	<p>Making a new branch (type without &lt;&lt; &gt;&gt;):</p>
	<blockquote>
		<p>- git checkout -b &lt;&lt;new_branch_name&gt;&gt;</p>
	</blockquote>
	<p>For safety, always check the status of your current branch first:</p>
	<blockquote>
		<p>- git status</p>
	</blockquote>
	<p>Only the files you have edited should be red. If other files are also red, make sure there is nothing wrong with those files.</p>
	<p>Adding code to a commit (type without &lt;&lt; &gt;&gt;):</p>
	<blockquote>
		<p>- git add &lt;&lt;filename.fileextension&gt;&gt;</p>
	</blockquote>
	<p>If you wish to add every change you made to the commit, simply type:</p>
	<blockquote>
		<p>- git add *</p>
	</blockquote>
	<p>Next, commit your added changes accompanied by a short description of the changes (include the quotation marks):</p>
	<blockquote>
		<p>- git commit -m "your short description of the changes"</p>
	</blockquote>
	<p>Once git accepts your commit (no issues), you can now push it upstream (so other people can pull your changes):</p>
	<blockquote>
		<p>- git push </p>
	</blockquote>
	<p>The very first time you wish to push to a branch, you must couple it to the remote branch for git to establish the connection.</p>
	<p>This goes as follows (type without &lt;&lt; &gt;&gt;):</p>
	<blockquote>
		<p>- git push --set-upstream origin &lt;&lt;branch_name&gt;&gt;</p>
	</blockquote>
	<p>You will only need to do this once per branch.</p>
  <p>Always ensure you are in the correct branch, and the status of your branch is caught up before committing or pushing.</p>
	<p>For safety reasons, it is not possible to push directly to the main branch.</p>
	<p>When committing changes, make sure you are in a branch that is relevant to those changes.</p>
  <p>If not, make one.</p>
</body>
</html>
