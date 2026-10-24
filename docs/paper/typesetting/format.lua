-- Typesetting-only transformations. Manuscript wording is left intact.
function Code(code)
  if #code.text > 24 and code.text:match('^[0-9a-f]+$') then
    return pandoc.RawInline('latex', '\\texttt{\\nolinkurl{' .. code.text .. '}}')
  end
  return code
end

function Pandoc(doc)
  local out = pandoc.List()
  local references = false
  for _, block in ipairs(doc.blocks) do
    if block.t == 'Header' then
      local title = pandoc.utils.stringify(block.content)
      if title == 'References' or title == '参考文献' then
        references = true
        out:insert(pandoc.RawBlock('latex', '\\needspace{8\\baselineskip}'))
        out:insert(block)
        out:insert(pandoc.RawBlock('latex', '\\small\\setlength{\\parskip}{4pt}\\setlength{\\emergencystretch}{2em}'))
      elseif title:match('^3%.6%.') then
        out:insert(pandoc.RawBlock('latex', '\\needspace{25\\baselineskip}'))
        out:insert(block)
      else
        out:insert(block)
      end
    elseif block.t == 'Para' and #block.content == 1 and block.content[1].t == 'Math' and block.content[1].mathtype == 'DisplayMath' then
      out:insert(pandoc.RawBlock('latex', '\\nopagebreak[4]'))
      out:insert(block)
    elseif block.t == 'CodeBlock' then
      out:insert(pandoc.RawBlock('latex', '\\begin{minipage}{\\linewidth}'))
      out:insert(block)
      out:insert(pandoc.RawBlock('latex', '\\end{minipage}'))
    elseif references and block.t == 'Para' then
      out:insert(pandoc.RawBlock('latex', '\\needspace{3\\baselineskip}'))
      out:insert(block)
    else
      out:insert(block)
    end
  end
  doc.blocks = out
  return doc
end
