[CmdletBinding()]
param([string]$EvidenceDirectory='')
$ErrorActionPreference='Stop'
if(-not $env:MOE4ALL_WORK_ROOT -or -not $env:MOE4ALL_WIZARD_TEST_TMP){throw 'Task work and tmp paths are required.'}
$repoRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'));$dataDir=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP 'unused';$script:IsPackagedInstall=$false
. (Join-Path $PSScriptRoot 'infr-dashboard.ps1')
. (Join-Path $PSScriptRoot 'infr-dashboard-state.ps1')
Initialize-DashboardRuntime
$code=@'
using System;using System.IO;using System.Drawing;using System.Collections;using System.Collections.Generic;using System.Reflection;using System.Windows.Forms;using MoE4All.Launcher;
public static class TaskFlowChecks {
 [System.Runtime.InteropServices.DllImport("user32.dll")] static extern IntPtr SendMessage(IntPtr hwnd,int message,IntPtr wparam,IntPtr lparam);
 static readonly BindingFlags Private=BindingFlags.Instance|BindingFlags.NonPublic;
 static int checks;
 static void Check(bool ok,string message){checks++;if(!ok)throw new Exception(message);}
 static Control Find(Control root,string name){var found=root.Controls.Find(name,true);return found.Length==0?null:found[0];}
 static IEnumerable<Control> Descendants(Control root){foreach(Control c in root.Controls){yield return c;foreach(var child in Descendants(c))yield return child;}}
 static void Styles(Control root){foreach(Control c in Descendants(root)){Check(!(c is TabControl),"No stock tab control remains");var text=c as TextBox;if(text!=null)Check(text.BorderStyle==BorderStyle.None&&(text.Parent is WizardInputFrame||text.Parent is WizardComboBox),"Padded input missing: "+text.Name);if(WizardTheme.Current.Dark&&c.Visible&&c.BackColor.A>0)Check(c.BackColor.R<220||c.BackColor.G<220||c.BackColor.B<220,"White background: "+c.Name);}}
 static void Reachable(Control control){Check(control!=null&&control.Visible&&control.Enabled,"Unreachable action");var center=new Point(control.Left+control.Width/2,control.Top+control.Height/2);var hit=control.Parent.GetChildAtPoint(center,GetChildAtPointSkip.Invisible|GetChildAtPointSkip.Disabled);Check(hit==control,"Action covered: "+control.Name+" by "+(hit==null?"none":hit.Name));}
 static void Choices(WizardComboBox combo){int original=combo.SelectedIndex;if(combo.Items.Count==0)return;var point=new IntPtr(((combo.Height/2)<<16)|(combo.Width-15));SendMessage(combo.Handle,0x0201,new IntPtr(1),point);SendMessage(combo.Handle,0x0202,IntPtr.Zero,point);var menu=(ContextMenuStrip)typeof(WizardComboBox).GetField("choices",Private).GetValue(combo);Check(menu!=null&&menu.Visible,"Dropdown click did not open: "+combo.Name);int index=combo.Items.Count-1;menu.Items[index].PerformClick();combo.CloseChoices();Check(combo.SelectedIndex==index,"Dropdown did not commit: "+combo.Name);combo.SelectedIndex=original;}
 static void Capture(Form form,string path){if(path.Length==0)return;form.PerformLayout();form.Refresh();Application.DoEvents();var layout=new System.Text.StringBuilder();foreach(Control c in Descendants(form))layout.AppendLine(c.Name+" | "+c.GetType().Name+" | "+c.Visible+" | "+c.Bounds+" | parent="+(c.Parent==null?"":c.Parent.Name));File.WriteAllText(path+".layout.txt",layout.ToString());using(var bitmap=new Bitmap(form.Width,form.Height)){form.DrawToBitmap(bitmap,new Rectangle(0,0,bitmap.Width,bitmap.Height));bitmap.Save(path);}}
 static void Dialog(NativeWizard ui,Action action,string page,bool accept,string capture){
  Exception fault=null;bool visited=false;int ticks=0;
  using(var timer=new Timer{Interval=40}){
   timer.Tick+=delegate{Form form=(Form)typeof(NativeWizard).GetField("settings",Private).GetValue(ui);if(form==null){if(++ticks>150){fault=new Exception("No dialog opened: "+page);timer.Stop();}return;}timer.Stop();
    try{visited=true;foreach(Control caption in Descendants(form))if(caption.Name.StartsWith("Caption")&&caption.AccessibleRole==AccessibleRole.PushButton)Check(!caption.Visible,"Child dialog exposes caption buttons");var host=(SettingsPages)Find(form,"SettingsContent");Check(host!=null&&host.TabPages.Count==1&&host.SelectedTab.Name==page,"Wrong contextual page: "+page);
     var groups=(IEnumerable)typeof(NativeWizard).GetField("settingsGroups",Private).GetValue(ui);foreach(var group in groups)group.GetType().GetField("Expanded").SetValue(group,true);
     typeof(NativeWizard).GetMethod("FillSettings",Private).Invoke(ui,null);Application.DoEvents();Styles(form);foreach(string common in new[]{"model","device","setup_mode","context","think_mode","reasoning_effort","server_addr","server_parallel","speech_model","speech_lexicon"})Check(Find(form,common)==null,"Common field duplicated in advanced dialog: "+common);
     var auth=Find(form,"server_auth") as CheckBox;if(auth!=null){auth.Checked=true;var secret=(TextBox)Find(form,"server_api_key");secret.Text="PRIVATE_FIXTURE";auth.Checked=false;auth.Checked=true;Check(Object.ReferenceEquals(secret,Find(form,"server_api_key"))&&secret.Text=="PRIVATE_FIXTURE"&&secret.UseSystemPasswordChar,"Dependent input lost its value");}
     var options=new List<WizardComboBox>();foreach(Control c in Descendants(form))if(c is WizardComboBox&&c.Visible)options.Add((WizardComboBox)c);foreach(var option in options)if(option.Visible)Choices(option);
     var slots=Find(form,"server_parallel") as TextBox;if(slots!=null)slots.Text="3";
     var current=host.SelectedTab;if(current.AutoScrollMinSize.Height>current.ClientSize.Height){current.AutoScrollPosition=new Point(0,100);Check(current.AutoScrollPosition.Y<0,"Settings cannot scroll");typeof(NativeWizard).GetMethod("FillSettings",Private).Invoke(ui,null);((Control)form.AcceptButton).Focus();current.AutoScrollPosition=Point.Empty;foreach(Control card in current.Controls){if(card.Name.StartsWith("Group_")&&card.Visible){Check(card.Top==14,"Scrolling and changing options drifted the first card");break;}}}
     Capture(form,capture);if(accept)form.AcceptButton.PerformClick();else form.CancelButton.PerformClick();
    }catch(Exception ex){fault=ex;form.Close();}
   };timer.Start();action();timer.Stop();
  }
  if(fault!=null)throw fault;Check(visited,"Action did not open a dialog: "+page);
 }
 public static int Run(string state,string schema,string name,string evidence){checks=0;
  using(var ui=new NativeWizard(state,schema,"UI test",false,"")){
   ui.ShowInTaskbar=false;ui.StartPosition=FormStartPosition.Manual;ui.Location=new Point(-3000,-3000);ui.Show();ui.SelectTask(ui.StateValue("launch_mode"));Application.DoEvents();
   Check(Find(ui,"MoreSettings")==null,"Global More settings still present");Styles(ui);
   var select=Find(ui,"setup_mode") as WizardComboBox;if(select!=null&&select.Visible)Choices(select);
   var original=ui.StateValue("server_parallel");
   foreach(string actionName in new[]{"ModelOptions","RuntimeOptions","GenerationOptions","ApiOptions","CustomTest"}){
    var button=Find(ui,actionName) as Button;if(button==null||!button.Visible)continue;Reachable(button);
    string page=actionName=="ModelOptions"?"models":(actionName=="RuntimeOptions"||actionName=="GenerationOptions")?"runtime":actionName=="ApiOptions"?"api":"benchmark";
    string image=evidence.Length==0?"":Path.Combine(evidence,name+"-"+page+".png");Dialog(ui,delegate{button.PerformClick();},page,false,image);
    Check(ui.StateValue("server_parallel")==original,"Cancel changed settings");
   }
   if(Find(ui,"ApiOptions")!=null){Dialog(ui,delegate{((Button)Find(ui,"ApiOptions")).PerformClick();},"api",true,"");Check(ui.StateValue("server_auth")=="true","Done did not keep auth edits");}
   int toolIndex=0;foreach(string page in new[]{"reports","maintenance"}){int item=toolIndex++;Dialog(ui,delegate{((Button)Find(ui,"Tools")).PerformClick();var menu=(ContextMenuStrip)typeof(NativeWizard).GetField("toolsMenu",Private).GetValue(ui);menu.Items[item].PerformClick();},page,false,evidence.Length==0?"":Path.Combine(evidence,name+"-"+page+".png"));}
   Capture(ui,evidence.Length==0?"":Path.Combine(evidence,name+"-main.png"));
  }return checks;
 }
}
'@
$source=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP 'task-flow-checks.cs';[IO.File]::WriteAllText($source,$code,[Text.UTF8Encoding]::new($true))
Add-Type -Path $source -ReferencedAssemblies @([MoE4All.Launcher.NativeWizard].Assembly.Location,'System.Windows.Forms.dll','System.Drawing.dll')
$cases=@(
 @{name='chat';launch_mode='chat'},
 @{name='daily';launch_mode='chat';setup_mode='conservative'},
 @{name='dedicated';launch_mode='chat';setup_mode='aggressive'},
 @{name='server-chat';launch_mode='server';server_llm_enabled=$true;server_vision=$false;server_embedding=$false;server_transcription=$false},
 @{name='server-all';launch_mode='server';server_llm_enabled=$true;server_vision=$true;server_embedding=$true;server_transcription=$true;server_speech=$true},
 @{name='english-server';wizard_language='en';launch_mode='server';server_llm_enabled=$true;server_vision=$true;server_embedding=$true;server_transcription=$true},
 @{name='embedding';launch_mode='server';server_llm_enabled=$false;server_embedding=$true;server_transcription=$false},
 @{name='audio';launch_mode='server';server_llm_enabled=$false;server_embedding=$false;server_transcription=$true},
 @{name='modules';launch_mode='server';server_llm_enabled=$false;server_embedding=$true;server_transcription=$true;server_speech=$true},
 @{name='speech';launch_mode='server';server_llm_enabled=$false;server_embedding=$false;server_transcription=$false;server_speech=$true},
 @{name='quick';launch_mode='benchmark';bench_kind='quick'},
 @{name='standard';launch_mode='benchmark';bench_kind='standard'},
 @{name='full';launch_mode='benchmark';bench_kind='full'},
 @{name='manual';launch_mode='benchmark';bench_kind='manual'},
 @{name='manual-decode';launch_mode='benchmark';bench_kind='manual';manual_bench_kind='decode'},
 @{name='manual-mixed';launch_mode='benchmark';bench_kind='manual';manual_bench_kind='mixed'}
)
$total=0
foreach($theme in @('dark','light')){foreach($case in $cases){
 $saved=@{wizard_language='zh';wizard_theme=$theme;launch_mode='chat';model='fixture/model.gguf';vision_projector='fixture/vision.gguf';embedding_model='fixture/embedding.gguf';transcription_model='fixture/audio.bin';speech_model='fixture/speech.gguf';speech_lexicon='fixture/speech-assets';setup_mode='manual';configure_memory=$true;configure_diagnostics=$true;configure_sampling=$true;server_parallel='2';device='Vulkan1';context='30k'}
 foreach($key in $case.Keys){if($key -ne 'name'){$saved[$key]=$case[$key]}}
 $state=Get-DashboardInitialState -Saved $saved;$schema=Get-DashboardSchema -State $state -DeviceOptions @([pscustomobject]@{value='Vulkan1';label='Fixture GPU'})
 $name=$case.name+'-'+$theme
 $total+=[TaskFlowChecks]::Run(($state|ConvertTo-Json -Depth 20 -Compress),($schema|ConvertTo-Json -Depth 20 -Compress),$name,$EvidenceDirectory)
 Write-Output ('PASS contextual task '+$name)
}}
Write-Output ('Task flows: '+$total+' checks passed across '+($cases.Count*2)+' scenarios. Own forms and fixtures only; no GPU/network/uploads.')
