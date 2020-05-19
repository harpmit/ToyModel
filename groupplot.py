import matplotlib
import subprocess, os
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rc('font', family='Helvetica')
import matplotlib.patches as ptch
import numpy as np

#could also include "size" (int) and "color" (string) as keys
std_font = {'family':'helvetica','weight':'normal'}
bold_font = {'family':'helvetica','weight':'bold'}
def apply_std_format(axes, std_font = std_font, bold_font = bold_font,
                     title=None,xlabel=None,ylabel=None,xticks=None,yticks=None, tick_percision=1,
                     xminorticks=None,yminorticks=None,fontsize=None,linewidth=1,
                     xticklabels=None,yticklabels=None):
    if fontsize:
        std_font['size']=fontsize
        bold_font['size']=fontsize
    
    #Lines and border
    axes.tick_params(which='both',right=True,top=True,direction='in',width=linewidth)
    axes.tick_params(which='major',length=linewidth*4)
    axes.tick_params(which='minor',length=linewidth*2)
    [axes.spines[i].set_linewidth(linewidth) for i in axes.spines]

    # Title
    if title:
        axes.set_title(title,fontdict=bold_font,pad=-15)
    # x axis
    if not xticks and type(xticks) != list:
        xticks = axes.get_xticks()
        xticks = [round(i,tick_percision) for i in xticks]
    axes.set_xticks(xticks)
    if not xminorticks and type(xminorticks) != list:
        xminorticks = [(float(xticks[i])+float(xticks[i+1]))/2. for i in range(len(xticks)-1)]
    axes.set_xticks(xminorticks,minor=True)
    if xticklabels:
        axes.set_xticklabels(xticklabels,fontdict=bold_font)
    else:
        labels = [str(i) for i in xticks]
        axes.set_xticklabels(labels,fontdict=bold_font)

    if xlabel:
        axes.set_xlabel(xlabel,fontdict=bold_font)

    #yaxis
    if not yticks and type(yticks) != list:
        yticks = axes.get_yticks()
        yticks = [round(i,tick_percision) for i in yticks]
    axes.set_yticks(yticks)
    if not yminorticks and type(yminorticks) != list:
        yminorticks = [(float(yticks[i])+float(yticks[i+1]))/2. for i in range(len(yticks)-1)]
    axes.set_yticks(yminorticks,minor=True)
    if yticklabels:
        axes.set_yticklabels(yticklabels,fontdict=bold_font)
    else:
        labels = [str(i) for i in yticks]
        axes.set_yticklabels(labels,fontdict=bold_font)
    if ylabel:
        axes.set_ylabel(ylabel,fontdict=bold_font)

def format_legend(legend,edgecolor='black',
    boxstyle='square',linewidth=1,color=None,alpha = None):
    ###Takes the legend object which is returned by plt.legend(),ax.legend(),or fig.legend()
    ###Apply standard group formatting to it
    ### boxstyle can be 'round','square',etc..
    
    box = legend.legendPatch
    box.set_boxstyle(boxstyle)
    box.set_linewidth(linewidth)
    if color:
        box.set_color(color)
    if alpha:
        box.set_alpha(alpha)

    box.set_edgecolor(edgecolor)

def highlight_point(ax,xy,xy_inset,height,inset_height,xdim_ydim_scale=1,
                    linestyle = '--',color=[140/255.,140/255.,140/255.],
                    linewidth = 1):
    #Circles a particular point and provides a position to display an inset
    #The axes object to add the highlight to
    #xy: coordinates of the point to highlight, as a list
    #xy_inset: coordinates of where to place the inset figure
    #height: radius of the circle highlighting the point on the y axis(float)
    #inset_height: #radius of the circle for the inset on the y axis(float)
    #xdim_ydim_scale: the relative size of the x and y dimensions. Obtained from (xmax-xmin)/(ymax-ymin) as a float

    width = height*xdim_ydim_scale
    inset_width = inset_height*xdim_ydim_scale
    circle1 = ptch.Ellipse(xy,height=height,width=width,
                          linestyle=linestyle,color=[0,0,0,0],linewidth=linewidth)
    circle2 = ptch.Ellipse(xy_inset,height=inset_height,width=inset_width,       
                          linestyle=linestyle,color=[0,0,0,0],linewidth=linewidth)
    circle1.set_edgecolor(color)
    circle2.set_edgecolor(color)
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    def is_outside(ellipse_xy,ellipse_height,ellipse_width,xy):
        #Returns True if the point xy is outside the ellips, otherwise returns false
        ellipse_value_1 = ((float(xy[0])-float(ellipse_xy[0]))**2)/(float(ellipse_width)/2.)**2
        ellipse_value_2 = ((float(xy[1])-float(ellipse_xy[1]))**2)/(float(ellipse_height)/2.)**2

        if (ellipse_value_1+ellipse_value_2) > 1:
            return True
        return False

    #add line between patches
    #First get the x coords
    smaller = min([xy[0],xy_inset[0]])
    larger = max([xy[0],xy_inset[0]])
    x = np.arange(smaller,larger,float(larger-smaller)/100).tolist()

    #Now the y coords
    if (xy[0] < xy_inset[0] and xy[1] > xy_inset[1]) or (xy_inset[0] < xy[0] and xy_inset[1] > xy[1]):
        reverse = True
    else:
        reverse = False
    smaller = min([xy[1],xy_inset[1]])
    larger = max([xy[1],xy_inset[1]])
    y = np.arange(smaller,larger,float(larger-smaller)/100).tolist()
    if reverse:
        y.reverse()

    xy = [[i,ii] for i,ii in zip(x,y) if is_outside(xy,height,width,[i,ii])]
    xy = [i for i in xy if is_outside(xy_inset,inset_height,inset_width,i)]

    x = [i for i,ii in xy]
    y = [ii for i,ii in xy]
    ax.plot(x,y,color=color,linestyle=linestyle,linewidth=linewidth)

def plot_reaction_coordinate(ax,delta_Gs,
                             line_length = 1,intermediate_spacing = 1,
                             line_pad = 1, x_margin = 1, color='blue',
                             offsetx=0,offsety=0,labels = None,
                             linestyles= None, linealphas=None):
    ### Takes a list of delta G values and plots a reaction coordinate figure
    #ax: matplotlib axes object
    #delta Gs: list
    #line_pad: amount of space to put between the lines and the text above them (float)
    #offsetx: shifts the whole graph to the right or left (float)
    #offsetx: shifts the whole graph up or down (float)
    #linestypes: a list of strings specifying the linestyles to apply, in order
    font = {'family':'helvetica','weight':'bold','size':8}
    
    intermediates = [0]
    for step in delta_Gs:
        intermediates.append(step+intermediates[-1])
    
    horizontal_lines = []
    for counter,intermediate in enumerate(intermediates):
        start = counter*intermediate_spacing+counter*line_length+offsetx
        end = counter*intermediate_spacing+(counter+1)*line_length+offsetx
        horizontal_lines.append(np.array([[start,intermediate+offsety],
                                          [end,intermediate+offsety]]))
        
    connecting_lines = []
    if len(horizontal_lines) > 1:
        for line_idx in range(len(horizontal_lines[:-1])):
            startx = horizontal_lines[line_idx][1,0]
            starty = horizontal_lines[line_idx][1,1]
            endx = horizontal_lines[line_idx+1][0,0]
            endy = horizontal_lines[line_idx+1][0,1]
            connecting_lines.append(np.array([[startx,starty],
                                              [endx,endy]]))
    if not labels:
        labels = ['']*len(horizontal_lines)
    for line,label in zip(horizontal_lines,labels):
        ax.plot(line[:,0],line[:,1],color=color,linewidth=3)
        text_position = float(line[0,0]+line[1,0])/2.
        if len(label) > 0:
            ax.text(text_position,line[0,1]+line_pad,label,
                    ha='center',va='center',
                    fontdict = font)

    if linestyles:
        if len(linestyles) == len(connecting_lines):
            pass
        else:
            raise Exception('Wrong length string in linestyles')
    else:
        linestyles = ['dotted']*len(connecting_lines)
    if linealphas:
        if len(linealphas) == len(connecting_lines):
            pass
        else:
            raise Exception('Wrong length string in linealphas')
    else:
        linealphas = [1]*len(connecting_lines)

    for line,style,alpha in zip(connecting_lines,linestyles,linealphas):
        ax.plot(line[:,0],line[:,1],color=color,linestyle=style,alpha=alpha)
        
    
    xmin = horizontal_lines[0][0,0]-x_margin
    xmax = horizontal_lines[-1][1,0]+x_margin
    if offsetx < 0:
        xmax = xmax-offsetx
    else:
        xmin = xmin-offsetx

    ax.set_xlim([xmin,xmax])
    ax.set_xticks([])
    ax.set_xticks([],minor=True)