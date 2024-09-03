r"""
Author:         Julius Schultz
Date:           2024-05-14
Description:    The plotlib contains classes for easier plotting.

"""
import copy
import os
import pickle
from collections.abc import Iterable 
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class Axes2D:
    def __init__(self, x_label = "x", y_label = "f(x)") -> None:
        self.x_label = x_label
        self.x_scale = 'linear'
        self.xlim = [None, None]
        self.draw_xticks = True
        self.draw_xlabel = True

        self.y_label = y_label
        self.y_scale = 'linear'
        self.ylim = [None, None]
        self.draw_yticks = True
        self.draw_ylabel = True

        self.axis_equal = False
        self.plot_grid = True
        self.title = ''

        self.xticksList = None
        self.xticksLabelList = None
        self.yticksList = None
        self.yticksLabelList = None
    
    def set_2D_ax_properties(self, ax):
        
        ax.set_xscale(self.x_scale)
        if self.xlim[0] is not None or self.xlim[1] is not None:
            ax.set_xlim(left = self.xlim[0], right = self.xlim[1])
        
        ax.set_yscale(self.y_scale)
        if self.ylim[0] is not None or self.ylim[1] is not None:
            ax.set_ylim(bottom = self.ylim[0], top = self.ylim[1])

        ax.grid(self.plot_grid)

        ax.set_title(self.title)
               
        if self.draw_xlabel:
            ax.set_xlabel(self.x_label)
        else:
            ax.set_xlabel("")

        if self.draw_ylabel:
            ax.set_ylabel(self.y_label)
        else:
            ax.set_ylabel("")
        
        ax.tick_params(labelbottom = self.draw_xticks, labelleft = self.draw_yticks)

        if self.axis_equal:
            ax.axis('equal')

        if self.xticksList is not None:
            if len(self.xticksList) == len(self.xticksLabelList):
                ax.set_xticks(self.xticksList)
                ax.set_xticklabels(self.xticksLabelList)

        if self.yticksList is not None:
            if len(self.yticksList) == len(self.yticksLabelList):
                ax.set_yticks(self.yticksList)
                ax.set_yticklabels(self.yticksLabelList)

        return ax

class LinePlot(Axes2D):
    def __init__(self, x, y, x_label = "x", y_label = "f(x)", **kwargs):
        super().__init__(x_label, y_label)
        self.x = x
        self.y = y
        self.kwargs = kwargs

        # Default line properties
        self.linewidth = 1.5
        self.linestyle = '-'
        self.color_no = None
        self.default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Set custom Plot Style
        self.set_default_line_kwargs()
        
    def set_default_line_kwargs(self):

        if 'linestyle' not in self.kwargs:
            self.kwargs["linestyle"] = self.linestyle

        if 'linewidth' not in self.kwargs:
            self.kwargs["linewidth"] = self.linewidth

    def add_line(self, y, legend = ""):

        if self.y.ndim == 1:
            self.y = self.y.reshape(-1,1)
        
        y = y.reshape(-1,1)

        self.y = np.concatenate([self.y, y], axis = 1) 

    def plot(self, ax):
        self.set_default_line_kwargs()

        if self.color_no is not None:
            self.kwargs["color"] = self.default_colors[self.color_no]

        ax.plot(self.x, self.y, **self.kwargs)
        
        ax = self.set_2D_ax_properties(ax)

        return ax



class HistPlot(Axes2D):
    def __init__(self, y, x_label = 'x', y_label = 'f(x)', vline = None, **hist_kwarg):
        super().__init__(x_label, y_label)
        self.y = y
        self.hist_kwarg = hist_kwarg

        self.legend = '_'

        # Default hist properties
        self.bins = None
        self.density = False

        # vertical line
        self.vline = vline

        # Set custom Plot Style
        self.set_default_hist_kwargs()
        
    def set_default_hist_kwargs(self):

        if 'bins' not in self.hist_kwarg:
            self.hist_kwarg["bins"] = self.bins

        if 'density' not in self.hist_kwarg:
            self.hist_kwarg["density"] = self.density

    def plot(self, ax):
        self.set_default_hist_kwargs()

        ax.hist(self.y, **self.hist_kwarg)

        if self.vline is not None:
            i = 0
            for val in self.vline:
                if i == 0:
                    coloStr = 'black'
                    i = 1
                else:
                    coloStr = 'red'
                ax.axvline(x = val, color = coloStr, linestyle = '--')
        
        ax = self.set_2D_ax_properties(ax)

        return ax
    
class ImagePlot(Axes2D):
    def __init__(self, image, x_label = "x", y_label = "y", 
                 cbarLabel = None, **img_kwargs):
        super().__init__(x_label, y_label)
        self.image = image
        self.img_kwargs = img_kwargs
        self.cbarLabel = cbarLabel
        self.legend = ""

    def set_default_img_kwargs(self):

        if 'cmap' not in self.img_kwargs:
            self.img_kwargs["cmap"] = 'viridis'
        
        if 'aspect' not in self.img_kwargs:
            self.img_kwargs["aspect"] = 'auto'
        
        if 'origin' not in self.img_kwargs:
            self.img_kwargs["origin"] = 'lower'

        if 'barScale' in self.img_kwargs:

            vminVal = self.img_kwargs["barScale"][1]
            vmaxVal = self.img_kwargs["barScale"][2]
            if "log" == self.img_kwargs["barScale"][0]:
                self.img_kwargs["norm"] = LogNorm(vmin = vminVal, 
                                                vmax = vmaxVal)
            else:
                self.img_kwargs["vmin"] = vminVal
                self.img_kwargs["vmax"] = vmaxVal
            self.img_kwargs.pop('barScale')

        if 'gridBoundary' in self.img_kwargs:
            self.gridBoundary = self.img_kwargs["gridBoundary"]
            self.img_kwargs.pop('gridBoundary')
        else:
            self.gridBoundary = None

    def plot(self, ax):

        self.set_default_img_kwargs()

        cax = ax.imshow(self.image, **self.img_kwargs)

        if self.cbarLabel is not None:
            cbar = plt.colorbar(cax)
            cbar.set_label(self.cbarLabel)
       
        ax = self.set_2D_ax_properties(ax)

        if self.gridBoundary is not None:
            ax.set_xticks(np.arange(-0.5, self.image.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.image.shape[0], 1), minor=True)

            ax.grid(False)
            ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

            ax.tick_params(which='major', size=0)
        
        return ax
    
class PlotData():
    def __init__(self, plots, filename, savepath, stylesheet, fig_size, custom_fig, subplot_grid, col_sort):
        self.plots = plots
        self.filename = filename
        self.fig_size = fig_size
        self.custom_fig = custom_fig
        self.savepath = savepath
        self.stylesheet = stylesheet
        self.subplot_grid = subplot_grid
        self.col_sort = col_sort 

class Plotter():
    def __init__(self, save_path = None, stylesheet = None, save_format = "pdf", save_plot_data = True, open_saved_plot = True):

        self.save_path = save_path
        self.save_format = save_format
        self.stylesheet = stylesheet
        self.save_plot_data = save_plot_data
        self.open_saved_plot = open_saved_plot

    def plot(self, *plots, filename = None, fig_size = None, subplot_grid = None, custom_fig = None, col_sort = True):

        if filename is not None: filename = filename.replace(" ", "_")
        n_subplots = len(plots)

        # load stylesheet if specified
        if self.stylesheet is not None:
            styleSheetVar = os.path.dirname(os.path.abspath(__file__)) + "/mpl_stylesheets/" + self.stylesheet + ".mplstyle"
            plt.style.use(styleSheetVar)

        # Create figure and ax object
        if custom_fig is not None:
            fig = copy.deepcopy(custom_fig[0])
            ax = copy.deepcopy(custom_fig[1])
        else:
            fig, ax = create_figure(n_subplots, subplot_grid, fig_size, col_sort = col_sort)

        # Loop over subplots
        for i, subplot in enumerate(plots):

            has_legend = False
            
            # Loop over plots in subplot
            if not isinstance(subplot, Iterable): subplot = [subplot]
            for j, plot_item in enumerate(subplot):
                
                if hasattr(plot_item, 'x'):
                    if plot_item.x == []:
                        fig.delaxes(ax[i])
                        continue
                ax[i] = plot_item.plot(ax[i])

                if type(ax[i]) != tuple:
                    if not ax[i].get_legend_handles_labels() == ([], []): has_legend = True        
                    if has_legend: ax[i].legend()
                else:
                    if not ax[i][-2].get_legend_handles_labels() == ([], []): has_legend = True
                    if has_legend: ax[i][-2].legend(handles = ax[i][-1])

        plt.tight_layout()
        
        # Saving
        if self.save_path is None:
            plt.show()

        else:
            unique_filename = self.save_plot(self.save_path, filename, fig)
            if self.save_plot_data:
                data_filename = self.save_path + "/data_" +  unique_filename + ".pkl"
                plt_data = PlotData(plots, filename, self.save_path, self.stylesheet, fig_size, custom_fig, subplot_grid, col_sort)
                with open(data_filename, 'wb') as file:
                    pickle.dump(plt_data, file)

                self.create_plotfile(self.save_path, unique_filename, data_filename)

        plt.close()
        return fig, ax

    def save_plot(self, path, filename, fig):
        
        # Output Folder
        now = datetime.now()
        time_string = now.strftime("%H_%M_%S")
        
        if filename is None:
            name = "Plot"
        else:
            name = filename
        unique_filename = time_string + "_" + name

        if self.save_format == "pdf":
            plt.savefig(path + "/" +  unique_filename + ".pdf") 
        elif self.save_format == "png":
            plt.savefig(path + "/" +  unique_filename + ".png", format='png', dpi = 600)

        # Open PDF in vs code
        if self.open_saved_plot:
            if self.save_format == "latex":
                save_format = "pdf"
            else:
                save_format = self.save_format
            system_command = "code " + path + "/" +  unique_filename + "." + save_format
            os.system(system_command)

        return unique_filename
    
    def create_plotfile(self, path, filename, data_file_path):
        
        # Get package name:
        # folder = glob.glob("./src/*.egg-info")
        # file = folder[0] + "/top_level.txt"
        # with open(file) as f:
        #     pkg_name = f.read().replace('\n', '')

        f= open(path + "/" +  filename + ".py","w")
        f.write(f"from sky.plotlib import * \nimport numpy as np\nimport pickle \n \n# Load plot data \n")
        f.write(f"file_path = '{data_file_path}' \n")
        f.write("with open(file_path, 'rb') as file:\n")
        f.write("   plt_data = pickle.load(file) \n\n")
        f.write("save_path = plt_data.savepath \n")
        f.write("plots = plt_data.plots \n")
        f.write("stylesheet = plt_data.stylesheet \n")
        f.write("filename = plt_data.filename \n")
        f.write("fig_size = plt_data.fig_size \n")
        f.write("custom_fig = plt_data.custom_fig \n")
        f.write("col_sort = plt_data.col_sort \n")
        f.write("subplot_grid = plt_data.subplot_grid \n \n")

        f.write("# Recreate plot \n")
        f.write("folder_path = save_path + '/regenerated_plots' \n")
        f.write("isExist = os.path.exists(folder_path) \n")
        f.write("if not isExist: \n")
        f.write("   os.makedirs(folder_path)\n")
        f.write("plotter = Plotter(save_path = save_path + '/regenerated_plots', stylesheet= stylesheet, save_plot_data = False) \n")
        f.write("plotter.plot(*plots, filename = filename, fig_size = fig_size, custom_fig = custom_fig, subplot_grid = subplot_grid, col_sort = col_sort) \n")

def create_figure(n_subplots, subplot_grid, fig_size, col_sort = True):
        if fig_size is not None:
            fig_size = np.asarray(fig_size)
            fig_size = fig_size / 2.54      # translate cm to inch
        
        if subplot_grid is not None:

            if n_subplots != subplot_grid[0]*subplot_grid[1]:
                raise ValueError("The number of subplots and the user defined subplot grid are not matching")

            n_row = subplot_grid[0]
            n_col = subplot_grid[1]

            if fig_size is None:
                size = calc_fig_size(subplots= (n_row, n_col) )
            else:
                size = fig_size
            fig, ax = plt.subplots(n_row, n_col, figsize=size)

            if col_sort:
                ax = ax.T

            ax = ax.flatten()

        else:
            if n_subplots == 1:
                n_row = 1
                n_col = 1
                if fig_size is None:
                    size = calc_fig_size(subplots= (n_row, n_col) )
                else:
                    size = fig_size
                fig, ax = plt.subplots(n_row, n_col, figsize=size)

                ax = np.array([ax])

            elif n_subplots == 2:
                n_row = 1
                n_col = 2
                if fig_size is None:
                    size = calc_fig_size(subplots= (n_row, n_col) )
                else:
                    size = fig_size
                fig, ax = plt.subplots(n_row, n_col, figsize=size)

                if col_sort:
                    ax = ax.T
                ax = ax.flatten()

            elif n_subplots == 3:
                n_row = 1
                n_col = 3
                if fig_size is None:
                    size = calc_fig_size(subplots= (n_row, n_col) )
                else:
                    size = fig_size
                fig, ax = plt.subplots(n_row, n_col, figsize=size)

                if col_sort:
                    ax = ax.T

                ax = ax.flatten()

            elif (n_subplots > 4) & (n_subplots < 7):
                n_row = 2
                n_col = 3
                if fig_size is None:
                    size = calc_fig_size(subplots= (n_row, n_col) )
                else:
                    size = fig_size
                fig, ax = plt.subplots(n_row, n_col, figsize=size)

                if col_sort:
                    ax = ax.T

                ax = ax.flatten()

            else:
                n_col = int(np.ceil(np.sqrt(n_subplots)))
                n_row = n_col
                if fig_size is None:
                    size = calc_fig_size(subplots= (n_row, n_col) )
                else:
                    size = fig_size
                fig, ax = plt.subplots(n_row, n_col, figsize=size)

                if col_sort:
                    ax = ax.T

                ax = ax.flatten()

        return fig, ax

def calc_fig_size(subplots=(1, 1), width_pt = 450):

        # Convert from pt to inches
        inches_per_pt = 1 / 72.27

        # Golden ratio to set aesthetic figure height
        # https://disq.us/p/2940ij3
        golden_ratio = (5**.5 - 1) / 2

        # Figure width in inches
        fig_width_in = width_pt * inches_per_pt
        # Figure height in inches
        fig_height_in = fig_width_in * golden_ratio

        return (fig_width_in * subplots[1], fig_height_in * subplots[0])

